"""Evidence-first bias diagnostics for the WC simulator.

Run: .venv/Scripts/python diagnostics.py

Writes output/diagnostics.json with out-of-sample slice calibration and
scoreline residuals. It does not change model parameters.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import rps
from wc_sim import ROOT, dc_grid, fit_model, load_matches, team_params

OUT = Path(__file__).parent / "output"
SCORELINES = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)]
CONFED_RULES = (
    ("AFC", re.compile(r"\bAFC\b|Asian Cup|WAFF|AFF|SAFF|EAFF", re.I)),
    ("CAF", re.compile(r"African|Africa Cup|CECAFA|COSAFA|WAFU|UNIFFAC", re.I)),
    ("CONCACAF", re.compile(r"CONCACAF|Gold Cup|Caribbean Cup|Copa Centroamericana|UNCAF", re.I)),
    ("CONMEBOL", re.compile(r"CONMEBOL|Copa Am[eé]rica", re.I)),
    ("OFC", re.compile(r"\bOFC\b|Oceania|Pacific Games", re.I)),
    ("UEFA", re.compile(r"\bUEFA\b|Euro qualification|European Championship|Nations League", re.I)),
)


def confed_for_tournament(tournament: str):
    for confed, pattern in CONFED_RULES:
        if pattern.search(str(tournament)):
            return confed
    return None


def infer_confederations(matches: pd.DataFrame):
    votes = defaultdict(Counter)
    for row in matches.dropna(subset=["home_score", "away_score"]).itertuples():
        confed = confed_for_tournament(row.tournament)
        if confed:
            votes[row.home_team][confed] += 1
            votes[row.away_team][confed] += 1
    meta = {}
    for team, counts in votes.items():
        confed, n = counts.most_common(1)[0]
        total = sum(counts.values())
        meta[team] = {"confed": confed, "evidence_matches": int(n),
                      "confidence": round(n / total, 3)}
    return meta


def _probs_from_grid(g):
    return np.array([np.tril(g, -1).sum(), np.trace(g), np.triu(g, 1).sum()])


def _forecast_records(df, confeds, start, refit_days, train_years, half_life, friendly_weight, verbose):
    df = df.copy()
    df["outcome"] = np.sign(df["away_score"] - df["home_score"]).map({-1: 0, 0: 1, 1: 2})
    start, end = pd.Timestamp(start), df["date"].max()
    records, skipped = [], 0
    block = start
    while block <= end:
        block_end = block + pd.Timedelta(days=refit_days)
        train = df[(df["date"] < block)
                   & (df["date"] >= block - pd.Timedelta(days=round(365.25 * train_years)))]
        test = df[(df["date"] >= block) & (df["date"] < block_end)]
        if test.empty:
            block = block_end
            continue
        atk, dfn, hfa, rho = team_params(fit_model(train, half_life, friendly_weight))
        for row in test.itertuples():
            if row.home_team not in atk or row.away_team not in atk:
                skipped += 1
                continue
            lam = math.exp(atk[row.home_team] + dfn[row.away_team] + (0.0 if row.neutral else hfa))
            mu = math.exp(atk[row.away_team] + dfn[row.home_team])
            g = dc_grid(lam, mu, rho)
            p = _probs_from_grid(g)
            y = int(row.outcome)
            hc = confeds.get(row.home_team, {}).get("confed", "UNKNOWN")
            ac = confeds.get(row.away_team, {}).get("confed", "UNKNOWN")
            records.append({
                "date": str(row.date.date()), "home_team": row.home_team, "away_team": row.away_team,
                "tournament": row.tournament, "kind": "friendly" if row.tournament == "Friendly" else "competitive",
                "neutral": bool(row.neutral), "home_confed": hc, "away_confed": ac,
                "confed_pair": f"{hc}-{ac}", "home_score": int(row.home_score),
                "away_score": int(row.away_score), "outcome": y,
                "p_home": float(p[0]), "p_draw": float(p[1]), "p_away": float(p[2]),
                "fav_p": float(p.max()), "fav_hit": int(p.argmax() == y),
                "rps": rps(p, y),
                "score_probs": {f"{x}-{z}": float(g[x, z]) for x, z in SCORELINES},
            })
        if verbose:
            print(f"diagnosed {block.date()} -> {block_end.date()}: train {len(train)}, test {len(test)}")
        block = block_end
    return records, skipped


def _mean(values):
    return round(float(np.mean(values)), 4) if values else None


def slice_summary(records, key, min_n=20):
    groups = defaultdict(list)
    for rec in records:
        groups[str(rec[key])].append(rec)
    out = []
    for name, rows in sorted(groups.items()):
        if len(rows) < min_n:
            continue
        out.append({
            key: name, "n": len(rows), "rps": _mean([r["rps"] for r in rows]),
            "favorite_pred": _mean([r["fav_p"] for r in rows]),
            "favorite_observed": _mean([r["fav_hit"] for r in rows]),
            "favorite_gap": round(_mean([r["fav_hit"] for r in rows]) - _mean([r["fav_p"] for r in rows]), 4),
            "draw_pred": _mean([r["p_draw"] for r in rows]),
            "draw_observed": _mean([r["outcome"] == 1 for r in rows]),
            "home_pred": _mean([r["p_home"] for r in rows]),
            "home_observed": _mean([r["outcome"] == 0 for r in rows]),
            "away_pred": _mean([r["p_away"] for r in rows]),
            "away_observed": _mean([r["outcome"] == 2 for r in rows]),
        })
    return out


def scoreline_summary(records):
    out = []
    for score in [f"{x}-{y}" for x, y in SCORELINES]:
        x, y = map(int, score.split("-"))
        pred = _mean([r["score_probs"][score] for r in records])
        obs = _mean([r["home_score"] == x and r["away_score"] == y for r in records])
        out.append({"score": score, "predicted": pred, "observed": obs,
                    "gap": round(obs - pred, 4), "n_observed": sum(
                        r["home_score"] == x and r["away_score"] == y for r in records)})
    return out


def optional_feature_coverage(root=ROOT):
    path = root / "data" / "match_features.csv"
    if not path.exists():
        return {"present": False, "note": "data/match_features.csv not found; xG/stats not used"}
    features = pd.read_csv(path)
    cols = [c for c in ["home_xg", "away_xg", "home_shots", "away_shots", "home_sot", "away_sot"] if c in features]
    return {"present": True, "rows": int(len(features)),
            "coverage": {c: round(float(features[c].notna().mean()), 3) for c in cols}}


def diagnose(start="2026-01-01", refit_days=45, train_years=4.0,
             half_life=550.0, friendly_weight=1.0, verbose=True):
    raw = pd.read_csv(ROOT / "data" / "matches.csv", parse_dates=["date"])
    df = load_matches(years=train_years + 1.5)
    confeds = infer_confederations(raw)
    records, skipped = _forecast_records(df, confeds, start, refit_days, train_years,
                                         half_life, friendly_weight, verbose)
    teams = set(df["home_team"]) | set(df["away_team"])
    known_confed = sum(t in confeds for t in teams)
    report = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"start": start, "refit_days": refit_days, "train_years": train_years,
                   "half_life": half_life, "friendly_weight": friendly_weight},
        "n": len(records), "skipped": skipped,
        "metrics": {
            "rps": _mean([r["rps"] for r in records]),
            "favorite_pred": _mean([r["fav_p"] for r in records]),
            "favorite_observed": _mean([r["fav_hit"] for r in records]),
            "draw_pred": _mean([r["p_draw"] for r in records]),
            "draw_observed": _mean([r["outcome"] == 1 for r in records]),
        },
        "data_coverage": {
            "teams_with_confed": known_confed,
            "teams_total": len(teams),
            "confed_coverage": round(known_confed / len(teams), 3) if teams else 0,
            "optional_match_features": optional_feature_coverage(),
        },
        "slices": {
            "kind": slice_summary(records, "kind"),
            "neutral": slice_summary(records, "neutral"),
            "home_confed": slice_summary(records, "home_confed"),
            "confed_pair": slice_summary(records, "confed_pair", min_n=12),
        },
        "scorelines": scoreline_summary(records),
        "notes": [
            "Confederations are inferred from confederation-specific competitions in the results data.",
            "Optional xG/stats are reported for coverage only and are not used unless coverage is adequate.",
        ],
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "diagnostics.json").write_text(json.dumps(report, indent=1))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--refit-days", type=int, default=45)
    ap.add_argument("--train-years", type=float, default=4.0)
    ap.add_argument("--half-life", type=float, default=550.0)
    ap.add_argument("--friendly-weight", type=float, default=1.0)
    args = ap.parse_args()
    r = diagnose(args.start, args.refit_days, args.train_years,
                 args.half_life, args.friendly_weight)
    print(f"diagnosed {r['n']} matches ({r['skipped']} skipped), RPS {r['metrics']['rps']}")
    print(f"confed coverage {r['data_coverage']['teams_with_confed']}/{r['data_coverage']['teams_total']}")
    print("largest scoreline gaps:")
    for row in sorted(r["scorelines"], key=lambda x: abs(x["gap"]), reverse=True)[:5]:
        print(f"  {row['score']}: predicted {row['predicted']:.3f}, observed {row['observed']:.3f}, gap {row['gap']:+.3f}")
    print("wrote output/diagnostics.json")


if __name__ == "__main__":
    main()
