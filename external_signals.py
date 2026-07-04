"""External team-strength priors derived from the generated player/market mart."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DEFAULT_EXTERNAL_WEIGHT = 0.12
MAX_RATE_ADJ = 0.25


def _z(values):
    s = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 2:
        return pd.Series(np.zeros(len(s)), index=s.index)
    filled = s.fillna(s.median())
    std = filled.std(ddof=0)
    if not std:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (filled - filled.mean()) / std


def build_external_strength(df: pd.DataFrame):
    market = _z(np.log1p(df["top23_market_value"].clip(lower=0)))
    rank = _z(-np.log(df["fifa_ranking"].clip(lower=1)))
    caps = _z(np.log1p(df["squad_caps"].clip(lower=0)))
    goals = _z(np.log1p(df["squad_goals"].clip(lower=0)))
    chemistry = _z(df["chemistry_score"]) if "chemistry_score" in df else 0.0
    raw = 0.52 * market + 0.25 * rank + 0.10 * caps + 0.05 * goals + 0.08 * chemistry
    strength = raw.clip(-2.5, 2.5)
    return {team: round(float(v), 4) for team, v in zip(df["team"], strength)}


def load_external_strength(path=ROOT / "output" / "external" / "project_team_enrichment.csv"):
    path = Path(path)
    if not path.exists():
        return {}, {"present": False, "note": "run external_data.py"}
    df = pd.read_csv(path)
    needed = {"team", "top23_market_value", "fifa_ranking", "squad_caps", "squad_goals"}
    missing = needed - set(df.columns)
    if missing:
        return {}, {"present": False, "note": f"external mart missing columns: {sorted(missing)}"}
    strength = build_external_strength(df)
    return strength, {"present": True, "rows": len(strength), "path": str(path)}


def external_rate_adjustment(team_a, team_b, strength, weight):
    if not strength or not weight:
        return 0.0
    diff = strength.get(team_a, 0.0) - strength.get(team_b, 0.0)
    return max(-MAX_RATE_ADJ, min(MAX_RATE_ADJ, float(weight) * diff))


def apply_external_prior(lam, mu, team_a, team_b, strength, weight):
    adj = external_rate_adjustment(team_a, team_b, strength, weight)
    if not adj:
        return lam, mu
    return lam * math.exp(adj), mu * math.exp(-adj)
