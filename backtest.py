"""Walk-forward calibration backtest + hyperparameter sweep for the Dixon-Coles engine.

Refits the model on a rolling basis (only data available before each block) and
scores out-of-sample 1X2 forecasts: RPS (primary), Brier, log-loss vs uniform and
train-frequency baselines, plus an in-sample vs out-of-sample gap (overfit check)
and favorite-calibration reliability bins. Single runs write output/backtest.json
(served to the web UI).

Run:  .venv/Scripts/python backtest.py [--start 2026-01-01] [--refit-days 45]
                                       [--half-life 550] [--friendly-weight 1.0]
Sweep: .venv/Scripts/python backtest.py --sweep   (grid over half-life x friendly weight)
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from external_signals import DEFAULT_EXTERNAL_WEIGHT, load_external_strength
from form_signals import DEFAULT_FORM_WEIGHT, build_recent_form_strength
from match_features import load_match_features
from wc_sim import (DEFAULT_GOAL_SCALE, DEFAULT_SCORELINE_DISPERSION, dc_grid,
                    fit_model, load_matches, match_rates, team_params)

OUT = Path(__file__).parent / "output"


def rps(probs, outcome):
    """Ranked probability score over ordered outcomes [home, draw, away]; lower is better."""
    c, o = np.cumsum(probs), np.zeros(3)
    o[outcome] = 1
    return float(np.sum((c[:2] - np.cumsum(o)[:2]) ** 2) / 2)


def grid_for_row(atk, dfn, hfa, rho, row, goal_scale=DEFAULT_GOAL_SCALE,
                 scoreline_dispersion=DEFAULT_SCORELINE_DISPERSION,
                 external_strength=None, external_weight=0.0,
                 form_strength=None, form_weight=0.0):
    venue = "" if row.neutral else row.home_team
    lam, mu = match_rates(atk, dfn, hfa, row.home_team, row.away_team, venue,
                          goal_scale, external_strength, external_weight,
                          form_strength, form_weight)
    return lam, mu, dc_grid(lam, mu, rho, scoreline_dispersion=scoreline_dispersion)


def outcome_probs(atk, dfn, hfa, rho, row, goal_scale=DEFAULT_GOAL_SCALE,
                  scoreline_dispersion=DEFAULT_SCORELINE_DISPERSION,
                  external_strength=None, external_weight=0.0,
                  form_strength=None, form_weight=0.0):
    _, _, g = grid_for_row(atk, dfn, hfa, rho, row, goal_scale, scoreline_dispersion,
                           external_strength, external_weight,
                           form_strength, form_weight)
    return np.array([np.tril(g, -1).sum(), np.trace(g), np.triu(g, 1).sum()])


def _score_rows(rows, atk, dfn, hfa, rho, sink, goal_scale=DEFAULT_GOAL_SCALE,
                scoreline_dispersion=DEFAULT_SCORELINE_DISPERSION,
                external_strength=None, external_weight=0.0,
                form_strength=None, form_weight=0.0):
    for row in rows.itertuples():
        if row.home_team not in atk or row.away_team not in atk:
            sink["skipped"] += 1
            continue
        lam, mu, g = grid_for_row(atk, dfn, hfa, rho, row, goal_scale, scoreline_dispersion,
                                  external_strength, external_weight,
                                  form_strength, form_weight)
        p = np.array([np.tril(g, -1).sum(), np.trace(g), np.triu(g, 1).sum()])
        y = int(row.outcome)
        sink["rps"].append(rps(p, y))
        sink["brier"].append(float(np.sum((p - np.eye(3)[y]) ** 2)))
        sink["logloss"].append(-math.log(max(p[y], 1e-12)))
        sink["fav_p"].append(float(p.max()))
        sink["fav_hit"].append(int(p.argmax() == y))
        sink["uniform"].append(rps(np.full(3, 1 / 3), y))
        sink["freq"].append(rps(sink["_freq"], y))
        total = int(row.home_score + row.away_score)
        top = sorted(((float(g[x, z]), x, z) for x in range(g.shape[0]) for z in range(g.shape[1])),
                     reverse=True)[:3]
        score_p = float(g[int(row.home_score), int(row.away_score)]) \
            if row.home_score < g.shape[0] and row.away_score < g.shape[1] else 0.0
        sink["pred_goals"].append(float(lam + mu))
        sink["actual_goals"].append(total)
        sink["pred_over25"].append(float(g[np.add.outer(np.arange(g.shape[0]), np.arange(g.shape[1])) > 2].sum()))
        sink["actual_over25"].append(1.0 if total > 2 else 0.0)
        sink["scoreline_logloss"].append(-math.log(max(score_p, 1e-12)))
        sink["modal_score_prob"].append(float(top[0][0]))
        sink["score_top1"].append(int((top[0][1], top[0][2]) == (row.home_score, row.away_score)))
        sink["score_top3"].append(int(any((x, z) == (row.home_score, row.away_score) for _, x, z in top)))
        sink["top_low_score"].append(int(top[0][1] + top[0][2] <= 2))


def run_backtest(df, start, refit_days, train_years, half_life, friendly_weight,
                 goal_scale=DEFAULT_GOAL_SCALE, external_strength=None,
                 scoreline_dispersion=DEFAULT_SCORELINE_DISPERSION,
                 external_weight=0.0, form_weight=DEFAULT_FORM_WEIGHT,
                 features=None, verbose=True):
    start, end = pd.Timestamp(start), df["date"].max()
    metric_keys = ("rps", "brier", "logloss", "fav_p", "fav_hit", "uniform", "freq",
                   "pred_goals", "actual_goals", "pred_over25", "actual_over25",
                   "scoreline_logloss", "modal_score_prob", "score_top1", "score_top3", "top_low_score")
    oos = {k: [] for k in metric_keys} | {"skipped": 0}
    ins = {k: [] for k in metric_keys} | {"skipped": 0}
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
        freq = train["outcome"].value_counts(normalize=True).reindex([0, 1, 2]).fillna(0).to_numpy()
        oos["_freq"] = ins["_freq"] = freq
        form_strength, _ = build_recent_form_strength(train, as_of=block, features=features,
                                                      external_strength=external_strength)
        _score_rows(test, atk, dfn, hfa, rho, oos, goal_scale, scoreline_dispersion,
                    external_strength, external_weight,
                    form_strength, form_weight)
        # in-sample slice: most recent train window of the same width (overfit gauge)
        _score_rows(train[train["date"] >= block - pd.Timedelta(days=refit_days)],
                    atk, dfn, hfa, rho, ins, goal_scale, scoreline_dispersion,
                    external_strength, external_weight,
                    form_strength, form_weight)
        if verbose:
            print(f"block {block.date()} -> {block_end.date()}: train {len(train)}, scored {len(test)}")
        block = block_end

    bins = []
    fav_p, fav_hit = np.array(oos["fav_p"]), np.array(oos["fav_hit"])
    for lo in np.arange(0.3, 1.0, 0.1):
        m = (fav_p >= lo) & (fav_p < lo + 0.1)
        if m.sum() >= 5:
            bins.append({"bin": f"{lo:.1f}-{lo + 0.1:.1f}", "n": int(m.sum()),
                         "predicted": round(float(fav_p[m].mean()), 3),
                         "observed": round(float(fav_hit[m].mean()), 3)})
    return {
        "config": {"start": str(start.date()), "refit_days": refit_days,
                   "train_years": train_years, "half_life": half_life,
                   "friendly_weight": friendly_weight, "goal_scale": goal_scale,
                   "scoreline_dispersion": scoreline_dispersion,
                   "external_weight": external_weight, "form_weight": form_weight},
        "n": len(oos["rps"]), "skipped": oos["skipped"],
        "rps": round(float(np.mean(oos["rps"])), 4),
        "rps_uniform": round(float(np.mean(oos["uniform"])), 4),
        "rps_trainfreq": round(float(np.mean(oos["freq"])), 4),
        "brier": round(float(np.mean(oos["brier"])), 4),
        "logloss": round(float(np.mean(oos["logloss"])), 4),
        "rps_in_sample": round(float(np.mean(ins["rps"])), 4) if ins["rps"] else None,
        "scoreline_calibration": {
            "predicted_goals": round(float(np.mean(oos["pred_goals"])), 3),
            "actual_goals": round(float(np.mean(oos["actual_goals"])), 3),
            "predicted_over25": round(float(np.mean(oos["pred_over25"])), 3),
            "actual_over25": round(float(np.mean(oos["actual_over25"])), 3),
            "exact_score_logloss": round(float(np.mean(oos["scoreline_logloss"])), 4),
            "modal_score_prob": round(float(np.mean(oos["modal_score_prob"])), 3),
            "top1_hit": round(float(np.mean(oos["score_top1"])), 3),
            "top3_hit": round(float(np.mean(oos["score_top3"])), 3),
            "top_low_score_share": round(float(np.mean(oos["top_low_score"])), 3),
        },
        "reliability": bins,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_backtest(start="2026-01-01", refit_days=45, train_years=4.0,
                   half_life=1100.0, friendly_weight=1.0,
                   goal_scale=DEFAULT_GOAL_SCALE,
                   scoreline_dispersion=DEFAULT_SCORELINE_DISPERSION,
                   external_weight=DEFAULT_EXTERNAL_WEIGHT,
                   form_weight=DEFAULT_FORM_WEIGHT, verbose=True):
    df = load_matches(years=train_years + 1.5)
    df["outcome"] = np.sign(df["away_score"] - df["home_score"]).map({-1: 0, 0: 1, 1: 2})
    external_strength, external_meta = load_external_strength(use_fiwc_impact=False)
    if not external_meta.get("present"):
        external_weight = 0.0
    features = load_match_features()
    r = run_backtest(df, start, refit_days, train_years, half_life, friendly_weight,
                     goal_scale, external_strength, scoreline_dispersion, external_weight, form_weight,
                     features, verbose)
    r["external_prior"] = {**external_meta, "weight": external_weight}
    r["form_prior"] = {"weight": form_weight, "mode": "opponent-adjusted recent form",
                       "feature_rows": 0 if features.empty else len(features)}
    OUT.mkdir(exist_ok=True)
    (OUT / "backtest.json").write_text(json.dumps(r, indent=1))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--refit-days", type=int, default=45)
    ap.add_argument("--train-years", type=float, default=4.0)
    ap.add_argument("--half-life", type=float, default=1100.0)
    ap.add_argument("--friendly-weight", type=float, default=1.0)
    ap.add_argument("--goal-scale", type=float, default=DEFAULT_GOAL_SCALE)
    ap.add_argument("--scoreline-dispersion", type=float, default=DEFAULT_SCORELINE_DISPERSION)
    ap.add_argument("--external-weight", type=float, default=DEFAULT_EXTERNAL_WEIGHT)
    ap.add_argument("--form-weight", type=float, default=DEFAULT_FORM_WEIGHT)
    ap.add_argument("--sweep", action="store_true", help="grid-search half-life x friendly weight")
    args = ap.parse_args()

    if args.sweep:
        df = load_matches(years=args.train_years + 1.5)
        df["outcome"] = np.sign(df["away_score"] - df["home_score"]).map({-1: 0, 0: 1, 1: 2})
        external_strength, _ = load_external_strength(use_fiwc_impact=False)
        features = load_match_features()
        print("half-life | friendly-w | external-w | RPS     | logloss | in-sample RPS")
        best = None
        for hl in (250, 550, 1100):
            for fw in (0.3, 0.6, 1.0):
                for ew in (0.0, 0.03, 0.06, 0.10):
                    r = run_backtest(df, args.start, 45, args.train_years, hl, fw,
                                     args.goal_scale, external_strength, args.scoreline_dispersion, ew,
                                     args.form_weight, features, verbose=False)
                    print(f"{hl:9.0f} | {fw:10.1f} | {ew:10.2f} | {r['rps']:.4f}  | {r['logloss']:.4f}  | {r['rps_in_sample']:.4f}")
                    if best is None or r["rps"] < best["rps"]:
                        best = r
        c = best["config"]
        print(f"\nbest by out-of-sample RPS: half-life {c['half_life']}, "
              f"friendly weight {c['friendly_weight']}, external weight {c['external_weight']} "
              f"(RPS {best['rps']})")
        return

    r = write_backtest(args.start, args.refit_days, args.train_years,
                       args.half_life, args.friendly_weight, args.goal_scale, args.scoreline_dispersion,
                       args.external_weight, args.form_weight)
    print(f"\n=== Walk-forward backtest: {r['n']} matches scored ({r['skipped']} skipped) ===")
    print(f"RPS   model {r['rps']} | uniform {r['rps_uniform']} | train-freq {r['rps_trainfreq']}   (lower better)")
    print(f"Brier {r['brier']} | log-loss {r['logloss']} | in-sample RPS {r['rps_in_sample']} "
          f"(gap {r['rps'] - r['rps_in_sample']:+.4f}; large positive = overfit)")
    sc = r["scoreline_calibration"]
    print(f"goals predicted {sc['predicted_goals']} vs actual {sc['actual_goals']} | "
          f"O2.5 predicted {sc['predicted_over25']} vs actual {sc['actual_over25']} | "
          f"exact-score top1 {sc['top1_hit']:.1%}, top3 {sc['top3_hit']:.1%} | "
          f"low-score modal {sc['top_low_score_share']:.1%}")
    print(f"external prior weight {r['config']['external_weight']} | {r['external_prior']}")
    print(f"form prior weight {r['config']['form_weight']} | {r['form_prior']}")
    print("reliability (favorite):", *(f"\n  {b['bin']}: predicted {b['predicted']:.2f} "
                                       f"observed {b['observed']:.2f} (n={b['n']})" for b in r["reliability"]))
    print("\nwrote output/backtest.json")


if __name__ == "__main__":
    main()
