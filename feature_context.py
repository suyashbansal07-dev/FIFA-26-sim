"""Forward-safe match context derived only from earlier results."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _team_snapshot(history, date, window):
    if not history:
        return {"days_rest": np.nan, "ppg": np.nan, "gf": np.nan, "ga": np.nan, "seen": 0}
    recent = history[-window:]
    return {
        "days_rest": int((date - history[-1]["date"]).days),
        "ppg": float(np.mean([m["points"] for m in recent])),
        "gf": float(np.mean([m["gf"] for m in recent])),
        "ga": float(np.mean([m["ga"] for m in recent])),
        "seen": len(history),
    }


def add_forward_safe_context(matches: pd.DataFrame, window=5) -> pd.DataFrame:
    df = matches.sort_values(["date", "home_team", "away_team"]).copy()
    histories = {}
    rows = []
    for row in df.itertuples():
        home = _team_snapshot(histories.get(row.home_team, []), row.date, window)
        away = _team_snapshot(histories.get(row.away_team, []), row.date, window)
        rows.append({
            "home_days_rest": home["days_rest"],
            "away_days_rest": away["days_rest"],
            "home_ppg_recent": home["ppg"],
            "away_ppg_recent": away["ppg"],
            "home_gf_recent": home["gf"],
            "away_gf_recent": away["gf"],
            "home_ga_recent": home["ga"],
            "away_ga_recent": away["ga"],
            "home_matches_seen": home["seen"],
            "away_matches_seen": away["seen"],
        })
        if pd.notna(row.home_score) and pd.notna(row.away_score):
            hs, away_score = int(row.home_score), int(row.away_score)
            hp = 3 if hs > away_score else 1 if hs == away_score else 0
            ap = 3 if away_score > hs else 1 if away_score == hs else 0
            histories.setdefault(row.home_team, []).append(
                {"date": row.date, "gf": hs, "ga": away_score, "points": hp})
            histories.setdefault(row.away_team, []).append(
                {"date": row.date, "gf": away_score, "ga": hs, "points": ap})
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def rest_bucket(home_days, away_days):
    if pd.isna(home_days) or pd.isna(away_days):
        return "unknown"
    delta = home_days - away_days
    if delta >= 4:
        return "home_rest_edge"
    if delta <= -4:
        return "away_rest_edge"
    return "balanced"


def form_bucket(home_ppg, away_ppg):
    if pd.isna(home_ppg) or pd.isna(away_ppg):
        return "unknown"
    delta = home_ppg - away_ppg
    if delta >= 0.75:
        return "home_form_edge"
    if delta <= -0.75:
        return "away_form_edge"
    return "balanced"
