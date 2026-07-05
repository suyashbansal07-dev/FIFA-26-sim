"""Forward-safe recent-form priors for already-played matches."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DEFAULT_FORM_WEIGHT = 0.0
MAX_FORM_RATE_ADJ = 0.12


def _result_score(gf, ga):
    if gf > ga:
        return 1.0
    if gf == ga:
        return 0.5
    return 0.0


def _expected_score(team, opponent, external_strength):
    if not external_strength:
        return 0.5
    diff = float(external_strength.get(team, 0.0)) - float(external_strength.get(opponent, 0.0))
    return 1.0 / (1.0 + math.exp(-0.85 * diff))


def _feature_lookup(features):
    if features is None or getattr(features, "empty", True):
        return {}
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    lookup = {}
    for row in df.to_dict("records"):
        key = (row["date"], row["home_team"], row["away_team"])
        lookup[key] = row
    return lookup


def _xg_edge(feature, home, away, team_is_home):
    if not feature:
        return 0.0
    hx, ax = feature.get("home_xg"), feature.get("away_xg")
    if pd.isna(hx) or pd.isna(ax):
        return 0.0
    edge = float(hx) - float(ax)
    if not team_is_home:
        edge = -edge
    return float(np.clip(edge / 3.0, -1.0, 1.0))


def _stat_pressure_edge(feature, team_is_home):
    if not feature:
        return 0.0
    edges = []
    for col, scale in (("shots", 12.0), ("sot", 5.0), ("corners", 8.0), ("possession", 35.0)):
        hv, av = feature.get(f"home_{col}"), feature.get(f"away_{col}")
        if pd.isna(hv) or pd.isna(av):
            continue
        edge = (float(hv) - float(av)) / scale
        edges.append(float(np.clip(edge if team_is_home else -edge, -1.0, 1.0)))
    return float(np.mean(edges)) if edges else 0.0


def _z_map(values):
    teams = list(values)
    s = pd.Series([values[t] for t in teams], dtype="float64")
    if len(s) < 2:
        return {t: 0.0 for t in teams}
    std = s.std(ddof=0)
    if not std:
        return {t: 0.0 for t in teams}
    z = ((s - s.mean()) / std).clip(-2.0, 2.0)
    return {t: round(float(v), 4) for t, v in zip(teams, z)}


def build_recent_form_strength(matches, as_of=None, window=6, features=None,
                               external_strength=None, min_matches=2):
    """Opponent-adjusted form signal from matches strictly before as_of.

    Uses result residual versus a rank/market expectation, plus small goal/xG
    edges. Missing xG is neutral. Returned values are z-scored across teams.
    """
    if matches is None or matches.empty:
        return {}, {"present": False, "note": "no matches"}
    df = matches.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    if as_of is not None:
        df = df[df["date"] < pd.Timestamp(as_of)]
    if df.empty:
        return {}, {"present": False, "note": "no prior completed matches"}
    lookup = _feature_lookup(features)
    hist = {}
    for row in df.sort_values(["date", "home_team", "away_team"]).itertuples():
        date_key = row.date.date().isoformat()
        feat = lookup.get((date_key, row.home_team, row.away_team))
        hs, aw = int(row.home_score), int(row.away_score)
        pairs = ((row.home_team, row.away_team, hs, aw, True),
                 (row.away_team, row.home_team, aw, hs, False))
        for team, opp, gf, ga, is_home in pairs:
            score = _result_score(gf, ga)
            expected = _expected_score(team, opp, external_strength)
            gd_edge = float(np.clip((gf - ga) / 3.0, -1.0, 1.0))
            xg_edge = _xg_edge(feat, row.home_team, row.away_team, is_home)
            stat_edge = _stat_pressure_edge(feat, is_home)
            quality = (score - expected) + 0.12 * gd_edge + 0.18 * xg_edge + 0.08 * stat_edge
            hist.setdefault(team, []).append({"date": row.date, "quality": quality})
    raw = {}
    for team, rows in hist.items():
        recent = rows[-window:]
        if len(recent) < min_matches:
            continue
        ages = np.arange(len(recent) - 1, -1, -1)
        weights = 0.72 ** ages
        raw[team] = float(np.average([r["quality"] for r in recent], weights=weights))
    strength = _z_map(raw)
    return strength, {"present": True, "rows": len(strength), "window": window,
                      "as_of": str(pd.Timestamp(as_of).date()) if as_of is not None else None}


def form_rate_adjustment(team_a, team_b, strength, weight):
    if not strength or not weight:
        return 0.0
    diff = strength.get(team_a, 0.0) - strength.get(team_b, 0.0)
    return max(-MAX_FORM_RATE_ADJ, min(MAX_FORM_RATE_ADJ, float(weight) * diff))


def apply_form_prior(lam, mu, team_a, team_b, strength, weight):
    adj = form_rate_adjustment(team_a, team_b, strength, weight)
    if not adj:
        return lam, mu
    return lam * math.exp(adj), mu * math.exp(-adj)
