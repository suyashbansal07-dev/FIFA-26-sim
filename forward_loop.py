"""Forward forecast ledger and calibration loop.

Run: .venv/Scripts/python forward_loop.py

Records current unplayed fixture forecasts to output/forward_forecasts.jsonl and
scores old forecasts once results arrive. Forecasts recorded after a match date
are marked late and excluded from calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import rps
from match_features import FEATURE_COLS, feature_coverage, find_match_feature, load_match_features
from wc_sim import ROOT

OUT = Path(__file__).parent / "output"
LEDGER = OUT / "forward_forecasts.jsonl"
CALIBRATION = OUT / "forward_calibration.json"


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path, rows):
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _forecast_id(row):
    key = "|".join(str(row[k]) for k in ("recorded_at", "fixture_id", "home", "away", "match_date"))
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def record_payload_forecasts(payload, ledger_path=LEDGER, now=None):
    now = now or datetime.now(timezone.utc)
    existing = _read_jsonl(ledger_path)
    seen = {row["forecast_id"] for row in existing}
    added = []
    for fx in payload.get("fixtures", []):
        if fx.get("played"):
            continue
        row = {
            "recorded_at": now.isoformat(timespec="seconds"),
            "fixture_id": fx["id"],
            "match_date": fx["date"],
            "home": fx["home"],
            "away": fx["away"],
            "venue": fx["venue"],
            "p_home": fx["p_home"],
            "p_draw": fx["p_draw"],
            "p_away": fx["p_away"],
            "over25": fx["over25"],
            "model": {
                "generated": payload["meta"].get("generated"),
                "half_life_days": payload["meta"].get("half_life_days"),
                "friendly_weight": payload["meta"].get("friendly_weight"),
                "hfa": payload["meta"].get("hfa"),
                "rho": payload["meta"].get("rho"),
            },
        }
        row["forecast_id"] = _forecast_id(row)
        if row["forecast_id"] not in seen:
            existing.append(row)
            seen.add(row["forecast_id"])
            added.append(row)
    _write_jsonl(ledger_path, existing)
    return {"ledger_rows": len(existing), "added": len(added)}


def _find_result(matches, forecast):
    date = pd.Timestamp(forecast["match_date"])
    played = matches.dropna(subset=["home_score", "away_score"])
    near = played[(played["date"] - date).abs() <= pd.Timedelta(days=2)]
    hit = near[(near["home_team"] == forecast["home"]) & (near["away_team"] == forecast["away"])]
    if hit.empty:
        hit = near[(near["home_team"] == forecast["away"]) & (near["away_team"] == forecast["home"])]
    return None if hit.empty else hit.iloc[-1]


def _outcome(row, forecast):
    if row["home_team"] == forecast["home"]:
        home_score, away_score = int(row["home_score"]), int(row["away_score"])
    else:
        home_score, away_score = int(row["away_score"]), int(row["home_score"])
    if home_score > away_score:
        y = 0
    elif home_score == away_score:
        y = 1
    else:
        y = 2
    return y, home_score, away_score


def _bins(settled):
    out = []
    for lo in np.arange(0.3, 1.0, 0.1):
        rows = [r for r in settled if lo <= r["favorite_pred"] < lo + 0.1]
        if len(rows) >= 3:
            out.append({"bin": f"{lo:.1f}-{lo + 0.1:.1f}", "n": len(rows),
                        "predicted": round(float(np.mean([r["favorite_pred"] for r in rows])), 3),
                        "observed": round(float(np.mean([r["favorite_hit"] for r in rows])), 3)})
    return out


def settle_forward_forecasts(matches, ledger_path=LEDGER, out_path=CALIBRATION):
    all_rows = _read_jsonl(ledger_path)
    # the ledger keeps every refresh's forecast (history); score only the latest
    # PRE-MATCH forecast per fixture, else re-forecast fixtures dominate the metrics.
    # Forecasts recorded after the match date stay individually counted as late.
    all_rows.sort(key=lambda r: r["recorded_at"])
    valid, late = [], 0
    for r in all_rows:
        if pd.Timestamp(r["recorded_at"]).date() > pd.Timestamp(r["match_date"]).date():
            late += 1
        else:
            valid.append(r)
    revisions = len(valid) - len({r["fixture_id"] for r in valid})
    rows = list({r["fixture_id"]: r for r in valid}.values())
    features = load_match_features()
    settled, pending = [], 0
    for forecast in rows:
        result = _find_result(matches, forecast)
        if result is None:
            pending += 1
            continue
        probs = np.array([forecast["p_home"], forecast["p_draw"], forecast["p_away"]])
        y, home_score, away_score = _outcome(result, forecast)
        settled_row = {
            "forecast_id": forecast["forecast_id"],
            "fixture_id": forecast["fixture_id"],
            "home": forecast["home"],
            "away": forecast["away"],
            "score": f"{home_score}-{away_score}",
            "outcome": y,
            "rps": rps(probs, y),
            "brier": float(np.sum((probs - np.eye(3)[y]) ** 2)),
            "logloss": -math.log(max(float(probs[y]), 1e-12)),
            "favorite_pred": float(probs.max()),
            "favorite_hit": int(probs.argmax() == y),
        }
        feat = find_match_feature(features, forecast["match_date"], forecast["home"], forecast["away"])
        if feat:
            settled_row["has_match_features"] = True
            for side in ("home", "away"):
                for col in FEATURE_COLS:
                    settled_row[f"{side}_{col}"] = feat.get(f"{side}_{col}")
        else:
            settled_row["has_match_features"] = False
        settled.append(settled_row)
    report = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ledger_rows": len(all_rows),
        "revisions_excluded": revisions,
        "settled": len(settled),
        "pending": pending,
        "late_excluded": late,
        "metrics": {
            "rps": round(float(np.mean([r["rps"] for r in settled])), 4) if settled else None,
            "brier": round(float(np.mean([r["brier"] for r in settled])), 4) if settled else None,
            "logloss": round(float(np.mean([r["logloss"] for r in settled])), 4) if settled else None,
            "favorite_pred": round(float(np.mean([r["favorite_pred"] for r in settled])), 4) if settled else None,
            "favorite_observed": round(float(np.mean([r["favorite_hit"] for r in settled])), 4) if settled else None,
        },
        "reliability": _bins(settled),
        "match_features": feature_coverage(settled),
        "settled_rows": settled,
    }
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    return report


def update_forward_loop(payload, matches_path=ROOT / "data" / "matches.csv"):
    recorded = record_payload_forecasts(payload)
    report = settle_forward_forecasts(pd.read_csv(matches_path, parse_dates=["date"]))
    return {**recorded, "settled": report["settled"], "pending": report["pending"],
            "late_excluded": report["late_excluded"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", default=str(OUT / "state.json"))
    args = ap.parse_args()
    state = json.loads(Path(args.state).read_text())
    meta = update_forward_loop(state["payload"])
    print(f"ledger {meta['ledger_rows']} rows, added {meta['added']}, "
          f"settled {meta['settled']}, pending {meta['pending']}, late {meta['late_excluded']}")


if __name__ == "__main__":
    main()
