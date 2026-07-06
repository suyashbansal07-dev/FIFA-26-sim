"""External team-strength priors derived from the generated player/market mart."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DEFAULT_EXTERNAL_WEIGHT = 0.15
MAX_RATE_ADJ = 0.25


def _z(values):
    s = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan)
    observed = s.dropna()
    if len(observed) < 2:
        return pd.Series(np.zeros(len(s)), index=s.index)
    std = observed.std(ddof=0)
    if not std:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s - observed.mean()) / std).fillna(0.0)


def _num_col(df, col):
    if col not in df:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def build_external_strength(df: pd.DataFrame, use_fiwc_impact=True):
    df = df.copy()
    top23 = _num_col(df, "top23_market_value")
    top11 = _num_col(df, "top11_market_value")
    top1 = _num_col(df, "top1_market_value")
    top3 = _num_col(df, "top3_market_value")
    top_attacker = _num_col(df, "top_attacker_market_value")
    quality_value = top11.where(top11.notna(), top23)
    depth_value = (top23 - top11).where(top23.notna() & top11.notna())
    quality = _z(np.log1p(quality_value.clip(lower=0)))
    depth = _z(np.log1p(depth_value.clip(lower=0)))
    star_share = (top1 / top23.replace(0, np.nan)).clip(lower=0, upper=1)
    star = (0.60 * _z(np.log1p(top1.clip(lower=0)))
            + 0.25 * _z(np.log1p(top3.clip(lower=0)))
            + 0.15 * _z(star_share))
    attacker = _z(np.log1p(top_attacker.clip(lower=0)))
    fiwc_impact = (
        0.70 * _z(np.log1p(_num_col(df, "fiwc_impact_score").clip(lower=0)))
        + 0.30 * _z(np.log1p(_num_col(df, "fiwc_top_impact_score").clip(lower=0)))
    ) if use_fiwc_impact else pd.Series(np.zeros(len(df)), index=df.index)
    rank = _z(-np.log(_num_col(df, "fifa_ranking").clip(lower=1)))
    caps = _z(np.log1p(_num_col(df, "squad_caps").clip(lower=0)))
    goals = _z(np.log1p(_num_col(df, "squad_goals").clip(lower=0)))
    chemistry = _z(_num_col(df, "chemistry_score"))
    position = _z(_num_col(df, "position_balance"))
    same_club = _z(_num_col(df, "same_club_share"))
    if use_fiwc_impact:
        raw = (0.27 * quality + 0.12 * depth + 0.34 * rank + 0.05 * caps
               + 0.03 * goals + 0.05 * chemistry + 0.03 * position + 0.02 * same_club
               + 0.025 * star + 0.015 * attacker + 0.05 * fiwc_impact)
    else:
        raw = (0.29 * quality + 0.13 * depth + 0.35 * rank + 0.055 * caps
               + 0.03 * goals + 0.055 * chemistry + 0.03 * position + 0.02 * same_club
               + 0.025 * star + 0.015 * attacker)
    strength = raw.clip(-2.5, 2.5)
    return {team: round(float(v), 4) for team, v in zip(df["team"], strength) if pd.notna(team)}


def load_external_strength(path=ROOT / "output" / "external" / "project_team_enrichment.csv",
                           use_fiwc_impact=True):
    path = Path(path)
    if not path.exists():
        return {}, {"present": False, "note": "run external_data.py"}
    df = pd.read_csv(path)
    needed = {"team", "top23_market_value", "fifa_ranking", "squad_caps", "squad_goals"}
    missing = needed - set(df.columns)
    if missing:
        return {}, {"present": False, "note": f"external mart missing columns: {sorted(missing)}"}
    strength = build_external_strength(df, use_fiwc_impact=use_fiwc_impact)
    return strength, {"present": True, "rows": len(strength), "path": str(path),
                      "use_fiwc_impact": use_fiwc_impact}


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
