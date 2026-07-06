"""Current-tournament xG/stat momentum prior for live forecasts."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from form_signals import (_expected_score, _feature_lookup, _result_score,
                          _stat_pressure_edge, _xg_edge, _z_map)

ROOT = Path(__file__).parent
DEFAULT_LIVE_WEIGHT = 0.03
MAX_LIVE_RATE_ADJ = 0.10
WORLD_CUP_START = "2026-06-01"


def build_live_context_strength(matches, features=None, as_of=None, window=5, min_matches=1,
                                external_strength=None):
    """Forward-safe World Cup-only momentum from prior result residuals, xG and shot pressure."""
    if matches is None or matches.empty:
        return {}, {"present": False, "note": "no matches"}
    df = matches.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    if "tournament" in df:
        df = df[df["tournament"].eq("FIFA World Cup")]
    df = df[df["date"] >= pd.Timestamp(WORLD_CUP_START)]
    if as_of is not None:
        df = df[df["date"] < pd.Timestamp(as_of)]
    if df.empty:
        return {}, {"present": False, "note": "no completed current-tournament matches"}

    lookup = _feature_lookup(features)
    hist = {}
    xg_rows = 0
    stat_rows = 0
    for row in df.sort_values(["date", "home_team", "away_team"]).itertuples():
        date_key = row.date.date().isoformat()
        feat = lookup.get((date_key, row.home_team, row.away_team))
        hs, aw = int(row.home_score), int(row.away_score)
        pairs = ((row.home_team, row.away_team, hs, aw, True),
                 (row.away_team, row.home_team, aw, hs, False))
        for team, opp, gf, ga, is_home in pairs:
            score_edge = _result_score(gf, ga) - _expected_score(team, opp, external_strength)
            gd_edge = float(np.clip((gf - ga) / 3.0, -1.0, 1.0))
            xg_edge = _xg_edge(feat, row.home_team, row.away_team, is_home)
            stat_edge = _stat_pressure_edge(feat, is_home)
            quality = 0.40 * score_edge + 0.20 * gd_edge + 0.25 * xg_edge + 0.15 * stat_edge
            hist.setdefault(team, []).append({"date": row.date, "quality": quality})
        if feat:
            if pd.notna(feat.get("home_xg")) and pd.notna(feat.get("away_xg")):
                xg_rows += 1
            if any(pd.notna(feat.get(f"home_{c}")) and pd.notna(feat.get(f"away_{c}"))
                   for c in ("shots", "sot", "corners", "possession")):
                stat_rows += 1

    raw = {}
    for team, rows in hist.items():
        recent = rows[-window:]
        if len(recent) < min_matches:
            continue
        ages = np.arange(len(recent) - 1, -1, -1)
        weights = 0.78 ** ages
        raw[team] = float(np.average([r["quality"] for r in recent], weights=weights))
    strength = _z_map(raw)
    return strength, {
        "present": True,
        "rows": len(strength),
        "matches": int(len(df)),
        "window": window,
        "source": "current_world_cup_xg_stats_results",
        "opponent_adjusted": bool(external_strength),
        "xg_rows": xg_rows,
        "stat_rows": stat_rows,
        "as_of": str(pd.Timestamp(as_of).date()) if as_of is not None else None,
    }


def live_rate_adjustment(team_a, team_b, strength, weight):
    if not strength or not weight:
        return 0.0
    diff = strength.get(team_a, 0.0) - strength.get(team_b, 0.0)
    return max(-MAX_LIVE_RATE_ADJ, min(MAX_LIVE_RATE_ADJ, float(weight) * diff))


def apply_live_prior(lam, mu, team_a, team_b, strength, weight):
    adj = live_rate_adjustment(team_a, team_b, strength, weight)
    if not adj:
        return lam, mu
    return lam * math.exp(adj), mu * math.exp(-adj)
