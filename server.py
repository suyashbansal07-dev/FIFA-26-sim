"""Flask app: web UI + prediction API + refresh pipeline for the WC-2026 sim.

Run: .venv/Scripts/python server.py [--port 8026] [--sims 1000000] [--auto-refresh-hours 0.25]

Endpoints:
  GET  /               web/index.html
  GET  /api/data       full payload (meta, fixtures+cards, bracket probabilities, ratings)
  GET  /api/predict    ?home=X&away=Y[&venue=C]  Dixon-Coles card for any matchup
  GET  /api/sample     same args; sample one scoreline (pens flag on draws)
  POST /api/refresh    scrape latest results -> refit -> re-simulate
  GET/POST /api/backtest  read or recompute walk-forward validation
"""
import argparse
import gzip
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

import fetch_data
from availability import apply_availability
from backtest import write_backtest
from consensus import build_consensus
from external_signals import DEFAULT_EXTERNAL_WEIGHT, external_rate_adjustment, load_external_strength
from form_signals import (DEFAULT_FORM_WEIGHT, build_recent_form_strength,
                          fitted_team_strength, form_rate_adjustment)
from forward_loop import update_forward_loop
from live_signals import DEFAULT_LIVE_WEIGHT, build_live_context_strength, live_rate_adjustment
from match_features import load_match_features
from wc_sim import (DEFAULT_GOAL_SCALE, DEFAULT_SCORELINE_DISPERSION, DEFAULT_SIMS,
                    SAMPLERS, Simulator, dc_grid, fit_model, known_winners,
                    load_matches, markets, match_rates, resolved_fixtures, run_ensemble,
                    run_tournament, shootout_rates, team_params, verdict_bracket)

ROOT = Path(__file__).parent
STATE_FILE = ROOT / "output" / "state.json"
BACKTEST_FILE = ROOT / "output" / "backtest.json"
SAMPLES_FILE = ROOT / "output" / "param_samples.json"
MARKET_FILE = ROOT / "output" / "market_odds.json"
AVAILABILITY_FILE = ROOT / "data" / "availability.json"
EXTERNAL_DIR = ROOT / "output" / "external"
MATCHES_FILE = ROOT / "data" / "matches.csv"
FEATURES_FILE = ROOT / "data" / "match_features.csv"
SHOOTOUTS_FILE = ROOT / "data" / "shootouts.csv"
FORWARD_LEDGER = ROOT / "output" / "forward_forecasts.jsonl"
FORWARD_CALIBRATION = ROOT / "output" / "forward_calibration.json"
FORWARD_CALIBRATION_APPLIED = ROOT / "output" / "forward_calibration_applied.json"
MODEL_CODE_FILES = (
    "wc_sim.py", "external_signals.py", "form_signals.py", "live_signals.py",
    "availability.py", "match_features.py",
)
app = Flask(__name__, static_folder="web", static_url_path="")
STATE = {"payload": None, "params": None, "pens": {}, "samples": None,
         "external_strength": {}, "external_meta": {},
         "form_strength": {}, "form_meta": {},
         "live_strength": {}, "live_meta": {}}  # params = (atk, dfn, hfa, rho)
LOCK = threading.Lock()
BACKTEST_LOCK = threading.Lock()
CFG = {"sims": DEFAULT_SIMS, "half_life": 1100.0, "friendly_weight": 1.0,
       "goal_scale": DEFAULT_GOAL_SCALE, "external_weight": DEFAULT_EXTERNAL_WEIGHT,
       "scoreline_dispersion": DEFAULT_SCORELINE_DISPERSION,
       "form_weight": DEFAULT_FORM_WEIGHT, "live_weight": DEFAULT_LIVE_WEIGHT,
       "years": 4.0, "sampler": "antithetic"}  # 1100d: sweep-validated, smallest OOS gap
KNOB_RANGES = {"half_life": (100, 2000), "friendly_weight": (0.0, 1.0),
               "goal_scale": (0.8, 1.3), "external_weight": (0.0, 0.15),
               "scoreline_dispersion": (0.0, 0.3),
               "form_weight": (0.0, 0.08), "live_weight": (0.0, 0.06),
               "sims": (10_000, DEFAULT_SIMS)}


def _file_signature(path):
    path = Path(path)
    if not path.exists():
        return {"present": False}
    raw = path.read_bytes()
    return {"present": True, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _availability_signature(path=None):
    return _file_signature(path or AVAILABILITY_FILE)


def _model_input_signature():
    return {
        "matches": _file_signature(MATCHES_FILE),
        "shootouts": _file_signature(SHOOTOUTS_FILE),
        "features": _file_signature(FEATURES_FILE),
        "external": _file_signature(EXTERNAL_DIR / "project_team_enrichment.csv"),
        "availability": _availability_signature(),
        "samples": _file_signature(SAMPLES_FILE),
        "code": {
            str(name): _file_signature(Path(name) if Path(name).is_absolute() else ROOT / name)
            for name in MODEL_CODE_FILES
        },
    }


def _clean(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    return v


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    return _clean(v)


def _load_external_payload():
    path = EXTERNAL_DIR / "project_team_enrichment.csv"
    if not path.exists():
        return {"present": False, "note": "run external_data.py"}
    df = pd.read_csv(path)
    meta_path = EXTERNAL_DIR / "external_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    keep = ["team", "confederation", "fifa_ranking", "current_nt_players",
            "top11_market_value", "top23_market_value", "squad_caps", "squad_goals",
            "top_player", "top_player_position", "top1_market_value",
            "top3_market_value", "top_attacker_market_value",
            "chemistry_score", "position_balance", "same_club_share",
            "fiwc_player_appearances", "fiwc_minutes", "fiwc_player_goals",
            "fiwc_assists", "fiwc_yellow_cards", "fiwc_red_cards",
            "fiwc_impact_score", "fiwc_impact_player", "fiwc_impact_player_position",
            "fiwc_top_impact_score", "fiwc_top_impact_goals",
            "fiwc_top_impact_assists", "fiwc_top_impact_minutes",
            "fiwc_top_market_player", "fiwc_top_market_minutes",
            "fiwc_top_market_goals", "fiwc_top_market_assists",
            "fiwc_top_market_impact_score", "fiwc_top_market_usage_share",
            "fiwc_top11_usage_share",
            "fiwc_starts", "fiwc_captain_starts"]
    for col in keep:
        if col not in df:
            df[col] = None
    rows = [{k: _clean(v) for k, v in row.items()} for row in df[keep].to_dict("records")]
    rows.sort(key=lambda r: r.get("top23_market_value") or 0, reverse=True)
    return {"present": True, "meta": meta, "teams": rows}


def _attach_external(payload):
    external = _load_external_payload()
    payload["external"] = external
    if not external.get("present"):
        return payload
    by_team = {r["team"]: r for r in external["teams"]}
    for r in payload.get("ratings", []):
        e = by_team.get(r["team"])
        if e:
            r.update({
                "fifa_ranking": e.get("fifa_ranking"),
                "top23_market_value": e.get("top23_market_value"),
                "top_player": e.get("top_player"),
                "top1_market_value": e.get("top1_market_value"),
                "top_attacker_market_value": e.get("top_attacker_market_value"),
                "fiwc_impact_player": e.get("fiwc_impact_player"),
                "fiwc_impact_score": e.get("fiwc_impact_score"),
                "fiwc_top_market_player": e.get("fiwc_top_market_player"),
                "fiwc_top_market_usage_share": e.get("fiwc_top_market_usage_share"),
                "fiwc_top11_usage_share": e.get("fiwc_top11_usage_share"),
                "squad_caps": e.get("squad_caps"),
                "squad_goals": e.get("squad_goals"),
            })
    payload["meta"]["availability"] = STATE.get("availability_meta", {"present": False})
    payload["meta"]["external_data"] = {
        "present": True,
        "rows": len(external["teams"]),
        "source": external.get("meta", {}).get("source"),
        "generated": external.get("meta", {}).get("generated"),
        "include_usage": external.get("meta", {}).get("include_usage"),
        "model_weight": CFG["external_weight"],
        "strength_rows": len(STATE.get("external_strength") or {}),
        "form_weight": CFG["form_weight"],
        "form_strength_rows": len(STATE.get("form_strength") or {}),
        "live_weight": CFG["live_weight"],
        "live_strength_rows": len(STATE.get("live_strength") or {}),
    }
    return payload


def _load_form_strength(df=None):
    df = df if df is not None else load_matches(CFG["years"])
    features = load_match_features(ROOT)
    atk, dfn, _, _ = STATE["params"]
    return build_recent_form_strength(
        df,
        features=features,
        baseline_strength=fitted_team_strength(atk, dfn),
    )


def _load_live_strength(df=None, as_of=None, features=None):
    df = df if df is not None else load_matches(CFG["years"])
    features = features if features is not None else load_match_features(ROOT)
    return build_live_context_strength(
        df, features=features, as_of=as_of,
        external_strength=STATE.get("external_strength"))


def _external_team(team, external=None):
    external = external or _load_external_payload()
    if not external.get("present"):
        return None
    return next((r for r in external["teams"] if r["team"] == team), None)


def _pair_mask(df, home, away):
    return (((df["home_team"] == home) & (df["away_team"] == away))
            | ((df["home_team"] == away) & (df["away_team"] == home)))


def _orient_match_row(row, home, away):
    same = row["home_team"] == home and row["away_team"] == away
    home_score = row["home_score"] if same else row["away_score"]
    away_score = row["away_score"] if same else row["home_score"]
    return {
        "date": str(pd.Timestamp(row["date"]).date()),
        "home": home,
        "away": away,
        "dataset_home": row["home_team"],
        "dataset_away": row["away_team"],
        "home_score": _clean(home_score),
        "away_score": _clean(away_score),
        "score": None if pd.isna(home_score) or pd.isna(away_score) else f"{int(home_score)}-{int(away_score)}",
        "tournament": row.get("tournament"),
        "neutral": _clean(row.get("neutral")),
    }


def _find_case_result(home, away, date=None):
    if not MATCHES_FILE.exists():
        return {"status": "missing", "note": "data/matches.csv not found"}
    df = pd.read_csv(MATCHES_FILE, parse_dates=["date"])
    hit = df[_pair_mask(df, home, away)]
    if date:
        day = pd.Timestamp(date)
        hit = hit[(hit["date"] - day).abs() <= pd.Timedelta(days=2)]
    if hit.empty:
        return {"status": "not_found"}
    hit = hit.sort_values("date")
    played = hit.dropna(subset=["home_score", "away_score"])
    row = played.iloc[-1] if not played.empty else hit.iloc[-1]
    out = _orient_match_row(row, home, away)
    out["status"] = "played" if out["score"] else "pending"
    return _jsonable(out)


def _recent_team_results(team, n=6):
    if not MATCHES_FILE.exists():
        return []
    df = pd.read_csv(MATCHES_FILE, parse_dates=["date"]).dropna(subset=["home_score", "away_score"])
    hit = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("date").tail(n)
    rows = []
    for row in hit.to_dict("records"):
        same = row["home_team"] == team
        gf = int(row["home_score"] if same else row["away_score"])
        ga = int(row["away_score"] if same else row["home_score"])
        opp = row["away_team"] if same else row["home_team"]
        rows.append({"date": str(pd.Timestamp(row["date"]).date()), "opponent": opp,
                     "score": f"{gf}-{ga}", "venue": "home" if same else "away",
                     "tournament": row.get("tournament")})
    return rows


def _read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_json(path, data):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


def _apply_forward_calibration():
    report = _read_json(FORWARD_CALIBRATION) or {}
    policy = report.get("calibration_policy") or {}
    action = policy.get("action")
    if action not in ("reduce_prior_or_goal_confidence", "allow_slightly_more_prior_confidence"):
        return {"applied": False, "action": action or "none", "reason": policy.get("reason", "no adjustment")}
    report_id = report.get("generated")
    state = _read_json(FORWARD_CALIBRATION_APPLIED) or {}
    if state.get("report_generated") == report_id:
        return {"applied": False, "action": action, "reason": "already applied",
                "external_weight": CFG["external_weight"], "live_weight": CFG["live_weight"]}
    fallback = -0.01 if action == "reduce_prior_or_goal_confidence" else 0.01
    adjustments = policy.get("knob_adjustments") or {
        "external_weight": fallback,
        "live_weight": fallback / 2,
    }
    knobs, applied = {}, False
    for knob, delta in adjustments.items():
        if knob not in KNOB_RANGES:
            continue
        before = CFG[knob]
        after = _clamp(before + float(delta), *KNOB_RANGES[knob])
        knobs[knob] = {"before": round(before, 4), "after": round(after, 4),
                       "delta": round(float(delta), 4), "applied": after != before}
        if after != before:
            CFG[knob] = after
            applied = True
    result = {"applied": applied, "action": action, "knobs": knobs,
              "report_generated": report_id, "settled": policy.get("settled"),
              "external_weight": CFG["external_weight"], "live_weight": CFG["live_weight"]}
    if "external_weight" in knobs:
        result.update({"knob": "external_weight",
                       "before": knobs["external_weight"]["before"],
                       "after": knobs["external_weight"]["after"]})
    if applied:
        _write_json(FORWARD_CALIBRATION_APPLIED, {
            **result,
            "applied_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    else:
        result["reason"] = "at configured bound or no supported knobs"
    return result


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _latest_forward_case(home, away, date=None):
    rows = [r for r in _read_jsonl(FORWARD_LEDGER)
            if {r.get("home"), r.get("away")} == {home, away}]
    if date:
        rows = [r for r in rows
                if abs((pd.Timestamp(r["match_date"]) - pd.Timestamp(date)).days) <= 2]
    pre = [r for r in rows
           if pd.Timestamp(r["recorded_at"]).date() <= pd.Timestamp(r["match_date"]).date()]
    out = {"ledger_rows": len(rows), "pre_match_rows": len(pre)}
    if pre:
        row = sorted(pre, key=lambda r: r["recorded_at"])[-1]
        out["latest_pre_match"] = {k: row.get(k) for k in (
            "recorded_at", "fixture_id", "match_date", "home", "away", "venue",
            "p_home", "p_draw", "p_away", "over25", "model")}
    report = _read_json(FORWARD_CALIBRATION) or {}
    settled = [r for r in report.get("settled_rows", [])
               if {r.get("home"), r.get("away")} == {home, away}]
    if settled:
        out["settled"] = settled[-1]
    return _jsonable(out)


def _case_features(home, away, date=None):
    if not FEATURES_FILE.exists():
        return None
    from match_features import find_match_feature
    features = load_match_features(ROOT)
    if features.empty:
        return None
    day = date or pd.Timestamp.today().date().isoformat()
    feat = find_match_feature(features, day, home, away)
    return _jsonable(feat) if feat else None


def case_diagnostic(home, away, venue="", date=None):
    c = card(home, away, venue)
    result = _find_case_result(home, away, date)
    day = date or result.get("date")
    features = _case_features(home, away, day)
    forward = _latest_forward_case(home, away, day)
    external = _load_external_payload()
    home_ext, away_ext = _external_team(home, external), _external_team(away, external)
    notes = []
    if result.get("status") == "played":
        hs, as_ = result["home_score"], result["away_score"]
        if hs > as_:
            actual, current_p = home, c["p_home"]
        elif hs == as_:
            actual, current_p = "draw", c["p_draw"]
        else:
            actual, current_p = away, c["p_away"]
        result["actual_outcome"] = actual
        result["current_model_actual_p"] = current_p
    elif result.get("status") == "pending":
        notes.append("Result row exists but has no final score yet.")
    else:
        notes.append("No local result row found for this pair/date.")
    if features and pd.notna(features.get("home_xg")) and pd.notna(features.get("away_xg")) and result.get("status") == "played":
        xg_home, xg_away = features["home_xg"], features["away_xg"]
        xg_side = home if xg_home > xg_away else away if xg_away > xg_home else "level"
        if xg_side != result.get("actual_outcome"):
            notes.append(f"xG leaned {xg_side}, final score leaned {result.get('actual_outcome')}.")
    for team, ext in ((home, home_ext), (away, away_ext)):
        if ext is None:
            notes.append(f"No Transfermarkt national-team row for {team}.")
    external_adj = external_rate_adjustment(
        home, away, STATE.get("external_strength"), CFG["external_weight"])
    form_adj = form_rate_adjustment(
        home, away, STATE.get("form_strength"), CFG["form_weight"])
    live_adj = live_rate_adjustment(
        home, away, STATE.get("live_strength"), CFG["live_weight"])
    pre_match_form = _pre_match_form_prior(home, away, day)
    pre_match_live = _pre_match_live_prior(home, away, day)
    return _jsonable({
        "card": c,
        "result": result,
        "forward": forward,
        "features": features,
        "external": {"home": home_ext, "away": away_ext},
        "external_prior": {
            "weight": CFG["external_weight"],
            "home_strength": STATE.get("external_strength", {}).get(home),
            "away_strength": STATE.get("external_strength", {}).get(away),
            "log_rate_adjustment": round(external_adj, 4),
        },
        "form_prior": {
            "weight": CFG["form_weight"],
            "home_strength": STATE.get("form_strength", {}).get(home),
            "away_strength": STATE.get("form_strength", {}).get(away),
            "log_rate_adjustment": round(form_adj, 4),
            "meta": STATE.get("form_meta", {}),
        },
        "pre_match_form_prior": pre_match_form,
        "live_prior": {
            "weight": CFG["live_weight"],
            "home_strength": STATE.get("live_strength", {}).get(home),
            "away_strength": STATE.get("live_strength", {}).get(away),
            "log_rate_adjustment": round(live_adj, 4),
            "meta": STATE.get("live_meta", {}),
        },
        "pre_match_live_prior": pre_match_live,
        "total_log_rate_adjustment": round(external_adj + form_adj + live_adj, 4),
        "recent": {home: _recent_team_results(home), away: _recent_team_results(away)},
        "notes": notes,
    })


def _pre_match_form_prior(home, away, day, df=None, features=None):
    if not day:
        return None
    df = df if df is not None else load_matches(CFG["years"])
    features = features if features is not None else load_match_features(ROOT)
    strength, meta = build_recent_form_strength(
        df,
        as_of=day,
        features=features,
    )
    adj = form_rate_adjustment(home, away, strength, CFG["form_weight"])
    return _jsonable({
        "as_of": str(pd.Timestamp(day).date()),
        "weight": CFG["form_weight"],
        "home_strength": strength.get(home),
        "away_strength": strength.get(away),
        "log_rate_adjustment": round(adj, 4),
        "meta": meta,
    })


def _pre_match_live_prior(home, away, day, df=None, features=None):
    if not day:
        return None
    strength, meta = _load_live_strength(df=df, as_of=day, features=features)
    adj = live_rate_adjustment(home, away, strength, CFG["live_weight"])
    return _jsonable({
        "as_of": str(pd.Timestamp(day).date()),
        "weight": CFG["live_weight"],
        "home_strength": strength.get(home),
        "away_strength": strength.get(away),
        "log_rate_adjustment": round(adj, 4),
        "meta": meta,
    })


def card(home, away, venue=""):
    atk, dfn, hfa, rho = STATE["params"]
    lam, mu = match_rates(atk, dfn, hfa, home, away, venue, CFG["goal_scale"],
                          STATE.get("external_strength"), CFG["external_weight"],
                          STATE.get("form_strength"), CFG["form_weight"],
                          STATE.get("live_strength"), CFG["live_weight"])
    g = dc_grid(lam, mu, rho, scoreline_dispersion=CFG["scoreline_dispersion"])
    h, d, a, o25, top = markets(g)
    return {"home": home, "away": away, "venue": venue or "neutral",
            "lam": round(lam, 3), "mu": round(mu, 3),
            "p_home": round(h, 4), "p_draw": round(d, 4), "p_away": round(a, 4),
            "over25": round(o25, 4), "external_weight": CFG["external_weight"],
            "form_weight": CFG["form_weight"], "live_weight": CFG["live_weight"],
            "top": [{"score": s, "p": round(p, 4)} for p, s in top],
            "grid": [[round(v, 5) for v in row] for row in g]}


def _load_samples(df):
    """Bootstrap parameter samples (uncertainty.py), only if they match the current
    knobs and data date — stale samples silently fall back to the point estimate."""
    if not SAMPLES_FILE.exists():
        return None
    s = json.loads(SAMPLES_FILE.read_text())
    if (s.get("half_life") == CFG["half_life"]
            and s.get("friendly_weight") == CFG["friendly_weight"]
            and s.get("data_max_date") == str(df["date"].max().date())):
        return s
    return None


def _bracket_rows(probs):
    return sorted(
        ({"team": t, "qf": round(p[0], 4), "sf": round(p[1], 4),
          "final": round(p[2], 4), "champion": round(p[3], 4),
          "bronze": round(p[4], 4)}
         for t, p in probs.items()), key=lambda r: -r["champion"])


def _prediction_summary(verdict, bracket_rows):
    top = bracket_rows[0] if bracket_rows else {}
    support = (verdict or {}).get("champion_match_support")
    definitive = (verdict or {}).get("champion")
    favorite = top.get("team")
    return {
        "definitive_champion": definitive,
        "definitive_basis": "forced_pick_per_match_advance_probability",
        "definitive_support": round(support, 4) if isinstance(support, (int, float)) else support,
        "monte_carlo_favorite": favorite,
        "monte_carlo_basis": "all_path_title_probability",
        "monte_carlo_champion_probability": top.get("champion"),
        "champion_disagreement": bool(definitive and favorite and definitive != favorite),
    }


def _attach_prediction(payload):
    if payload:
        payload["prediction"] = _prediction_summary(payload.get("verdict"), payload.get("bracket", []))
    return payload


def refresh():
    """Scrape -> refit -> re-simulate -> rebuild payload. Serialized by LOCK."""
    with LOCK:
        calibration_applied = _apply_forward_calibration()
        fetch_meta = fetch_data.fetch(quiet=True)
        df = load_matches(CFG["years"])
        atk, dfn, hfa, rho = team_params(fit_model(df, CFG["half_life"], CFG["friendly_weight"]))
        STATE["params"] = (atk, dfn, hfa, rho)

        bracket = json.loads((ROOT / "bracket_2026.json").read_text())
        shootouts = pd.read_csv(SHOOTOUTS_FILE, parse_dates=["date"])
        STATE["pens"] = shootout_rates(shootouts)
        known = known_winners(bracket, df, shootouts)
        STATE["samples"] = _load_samples(df)
        STATE["external_strength"], STATE["external_meta"] = load_external_strength(EXTERNAL_DIR / "project_team_enrichment.csv")
        STATE["external_strength"], STATE["availability_meta"] = apply_availability(STATE["external_strength"], AVAILABILITY_FILE)
        STATE["form_strength"], STATE["form_meta"] = _load_form_strength(df)
        STATE["live_strength"], STATE["live_meta"] = _load_live_strength(df)
        if STATE["samples"]:
            probs, paths = run_ensemble(STATE["samples"]["samples"], STATE["pens"],
                                        bracket, known, CFG["sims"], CFG["sampler"],
                                        goal_scale=CFG["goal_scale"],
                                        scoreline_dispersion=CFG["scoreline_dispersion"],
                                        external_strength=STATE["external_strength"],
                                        external_weight=CFG["external_weight"],
                                        form_strength=STATE["form_strength"],
                                        form_weight=CFG["form_weight"],
                                        live_strength=STATE["live_strength"],
                                        live_weight=CFG["live_weight"])
            uncertainty = {"mode": "bootstrap-ensemble", "boots": STATE["samples"]["boots"]}
        else:
            sim = Simulator(atk, dfn, hfa, rho, np.random.default_rng(), pens=STATE["pens"],
                            goal_scale=CFG["goal_scale"],
                            scoreline_dispersion=CFG["scoreline_dispersion"],
                            external_strength=STATE["external_strength"],
                            external_weight=CFG["external_weight"],
                            form_strength=STATE["form_strength"],
                            form_weight=CFG["form_weight"],
                            live_strength=STATE["live_strength"],
                            live_weight=CFG["live_weight"])
            probs, paths = run_tournament(sim, bracket, known, CFG["sims"], CFG["sampler"],
                                          return_paths=True)
            uncertainty = {"mode": "point-estimate"}

        fixtures = []
        for fx in resolved_fixtures(bracket, known):
            c = card(fx["home"], fx["away"], fx["venue_country"])
            c.update(id=fx["id"], date=fx["date"], round=fx["round"],
                     played=fx["id"] in known, winner=known.get(fx["id"]))
            fixtures.append(c)
        verdict_sim = Simulator(atk, dfn, hfa, rho, np.random.default_rng(26), pens=STATE["pens"],
                                goal_scale=CFG["goal_scale"],
                                scoreline_dispersion=CFG["scoreline_dispersion"],
                                external_strength=STATE["external_strength"],
                                external_weight=CFG["external_weight"],
                                form_strength=STATE["form_strength"],
                                form_weight=CFG["form_weight"],
                                live_strength=STATE["live_strength"],
                                live_weight=CFG["live_weight"])
        bracket_rows = _bracket_rows(probs)
        verdict = verdict_bracket(verdict_sim, bracket, known)

        STATE["payload"] = {
            "meta": {**fetch_meta, "trained_matches": len(df),
                     "train_from": str(df["date"].min().date()),
                     "teams": df["home_team"].nunique(),
                     "hfa": round(hfa, 3), "rho": round(rho, 3),
                     "sims": CFG["sims"], "half_life_days": CFG["half_life"],
                     "friendly_weight": CFG["friendly_weight"], "goal_scale": CFG["goal_scale"],
                     "scoreline_dispersion": CFG["scoreline_dispersion"],
                     "external_weight": CFG["external_weight"],
                     "form_weight": CFG["form_weight"],
                     "live_weight": CFG["live_weight"],
                     "live_context": {**STATE.get("live_meta", {}), "weight": CFG["live_weight"]},
                     "sampler": CFG["sampler"],
                     "uncertainty": uncertainty,
                     "forward_calibration_applied": calibration_applied,
                     "availability_input": _availability_signature(),
                     "model_input_signature": _model_input_signature(),
                     "freshness": _freshness_meta(fetch_meta, bracket, known),
                     "generated": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            "fixtures": fixtures,
            "tree": {k: bracket[k] for k in ("qf", "sf", "final")},
            "known": known,
            "verdict": verdict,
            "bracket": bracket_rows,
            "prediction": _prediction_summary(verdict, bracket_rows),
            "teams": sorted(atk),
            "ratings": sorted(
                ({"team": t, "attack": round(atk[t], 3), "defence": round(dfn[t], 3)}
                 for t in atk), key=lambda r: -(r["attack"] - r["defence"]))[:30],
            "consensus": build_consensus(paths, bracket, known),
        }
        _attach_external(STATE["payload"])
        STATE["payload"]["meta"]["forward_loop"] = update_forward_loop(STATE["payload"])
        STATE_FILE.parent.mkdir(exist_ok=True)
        STATE_FILE.write_text(json.dumps(
            {"payload": STATE["payload"], "pens": STATE["pens"],
             "params": {"attack": atk, "defence": dfn, "hfa": hfa, "rho": rho}}))
    return STATE["payload"]["meta"]


def _state_compatible(payload):
    return bool(payload and payload.get("bracket")
                and all("bronze" in row for row in payload["bracket"])
                and payload.get("verdict")
                and "scoreline_dispersion" in payload.get("meta", {})
                and "live_weight" in payload.get("meta", {}))


def _utc_timestamp(value):
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _state_needs_refresh(meta, today=None, max_age_hours=None):
    if not meta.get("generated"):
        return True
    if meta.get("model_input_signature") != _model_input_signature():
        return True
    now = _utc_timestamp(today) if today is not None else pd.Timestamp.now(tz="UTC")
    generated = _utc_timestamp(meta["generated"])
    if generated.date() < now.date():
        return True
    if max_age_hours and max_age_hours > 0:
        return (now - generated).total_seconds() > max_age_hours * 3600
    return False


def _freshness_meta(fetch_meta, bracket, known, today=None):
    today = pd.Timestamp(today).date() if today is not None else pd.Timestamp.today().date()
    due = [fx["id"] for fx in resolved_fixtures(bracket, known)
           if fx["id"] not in known and pd.Timestamp(fx["date"]).date() < today]
    newest = fetch_meta.get("newest_result")
    lag = (today - pd.Timestamp(newest).date()).days if newest else None
    return {"today": today.isoformat(), "result_lag_days": lag,
            "overdue_unplayed_slots": due, "stale": bool(due)}


def load_state():
    if STATE_FILE.exists():
        s = json.loads(STATE_FILE.read_text())
        if not _state_compatible(s.get("payload")):
            return False
        STATE["payload"] = s["payload"]
        STATE["pens"] = s.get("pens", {})
        p = s["params"]
        STATE["params"] = (p["attack"], p["defence"], p["hfa"], p["rho"])
        df = load_matches(CFG["years"])
        STATE["samples"] = _load_samples(df)
        STATE["external_strength"], STATE["external_meta"] = load_external_strength(EXTERNAL_DIR / "project_team_enrichment.csv")
        STATE["external_strength"], STATE["availability_meta"] = apply_availability(STATE["external_strength"], AVAILABILITY_FILE)
        STATE["form_strength"], STATE["form_meta"] = _load_form_strength(df)
        STATE["live_strength"], STATE["live_meta"] = _load_live_strength(df)
        _attach_external(STATE["payload"])
        _attach_prediction(STATE["payload"])
        return True
    return False


API_TOKEN = os.environ.get("WC26_TOKEN")  # set to require Bearer auth on mutating endpoints


@app.before_request
def _guard_mutations():
    if API_TOKEN and request.method == "POST" \
            and request.headers.get("Authorization") != f"Bearer {API_TOKEN}":
        return jsonify({"error": "unauthorized"}), 401


@app.after_request
def _gzip_json(resp):
    if (resp.content_type or "").startswith("application/json") and not resp.direct_passthrough \
            and "gzip" in request.headers.get("Accept-Encoding", "") and resp.status_code == 200:
        body = resp.get_data()
        if len(body) > 2048:
            resp.set_data(gzip.compress(body, 6))
            resp.headers["Content-Encoding"] = "gzip"
            resp.headers["Content-Length"] = str(len(resp.get_data()))
    return resp


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/api/data")
def api_data():
    return jsonify(_attach_prediction(_attach_external(STATE["payload"])))


@app.get("/api/external")
def api_external():
    return jsonify(_load_external_payload())


@app.get("/api/case")
def api_case():
    args = _matchup_args()
    if not args:
        return jsonify({"error": "need distinct rated teams: ?home=X&away=Y[&venue=C][&date=YYYY-MM-DD]"}), 400
    return jsonify(case_diagnostic(*args, date=request.args.get("date")))


def _matchup_args():
    home, away = request.args.get("home"), request.args.get("away")
    atk = STATE["params"][0]
    if not home or not away or home not in atk or away not in atk or home == away:
        return None
    return home, away, request.args.get("venue", "")


@app.get("/api/predict")
def api_predict():
    args = _matchup_args()
    if not args:
        return jsonify({"error": "need distinct rated teams: ?home=X&away=Y[&venue=C]"}), 400
    return jsonify(card(*args))


@app.get("/api/sample")
def api_sample():
    args = _matchup_args()
    if not args:
        return jsonify({"error": "need distinct rated teams: ?home=X&away=Y[&venue=C]"}), 400
    home, away, venue = args
    if STATE["samples"]:  # sample parameter uncertainty too, not just scoreline noise
        rng0 = np.random.default_rng()
        ps = STATE["samples"]["samples"][int(rng0.integers(len(STATE["samples"]["samples"])))]
        atk, dfn, hfa, rho = ps["attack"], ps["defence"], ps["hfa"], ps["rho"]
    else:
        atk, dfn, hfa, rho = STATE["params"]
    sim = Simulator(atk, dfn, hfa, rho, np.random.default_rng(), pens=STATE["pens"],
                    goal_scale=CFG["goal_scale"],
                    scoreline_dispersion=CFG["scoreline_dispersion"],
                    external_strength=STATE.get("external_strength"),
                    external_weight=CFG["external_weight"],
                    form_strength=STATE.get("form_strength"),
                    form_weight=CFG["form_weight"],
                    live_strength=STATE.get("live_strength"),
                    live_weight=CFG["live_weight"])
    lam, mu, flat, _ = sim.grid_for(home, away, venue)
    x, y = divmod(int(sim.rng.choice(flat.size, p=flat)), 10)
    res = {"home": home, "away": away, "home_goals": x, "away_goals": y,
           "extra_time": None, "pens": None}
    if x == y:  # mirror Simulator.play: 30' Poisson ET, then historical shootout rates
        ex, ey = int(sim.rng.poisson(lam / 3)), int(sim.rng.poisson(mu / 3))
        res["extra_time"] = f"{x + ex}-{y + ey}"
        if ex == ey:
            res["pens"] = home if sim.rng.random() < sim.pens_prob(home, away) else away
    return jsonify(res)


def _apply_knobs(body):
    changed = {}
    for k in ("half_life", "friendly_weight", "goal_scale", "scoreline_dispersion",
              "external_weight", "form_weight", "live_weight", "sims"):
        if k in body:
            lo, hi = KNOB_RANGES[k]
            v = min(max(float(body[k]), lo), hi)
            CFG[k] = int(v) if k == "sims" else v
            changed[k] = CFG[k]
    if body.get("sampler") in SAMPLERS:
        CFG["sampler"] = body["sampler"]
        changed["sampler"] = CFG["sampler"]
    return changed


def _clamp(v, lo, hi):
    return min(max(v, lo), hi)


def _known_with_overrides(bracket, known, overrides, counterfactual=False):
    out = dict(known)
    if counterfactual:
        children = {}
        slots = bracket.get("qf", []) + bracket.get("sf", [])
        slots += [fx for fx in (bracket.get("final"), bracket.get("third_place")) if fx]
        for fx in slots:
            for parent in fx.get("from", []):
                children.setdefault(parent, set()).add(fx["id"])
        stack, drop = list(overrides), set()
        while stack:
            slot = stack.pop()
            for child in children.get(slot, ()):
                if child not in drop:
                    drop.add(child)
                    stack.append(child)
        for slot in drop:
            out.pop(slot, None)
    out.update(overrides)
    return out


def _validate_overrides(bracket, overrides, known, counterfactual=False):
    """Any determined, not-yet-played slot is pinnable (R16 now, QF/SF/F as they form).
    counterfactual=True additionally allows rewriting played slots ('what if X had won')."""
    basis = _known_with_overrides(bracket, known, overrides, counterfactual) if counterfactual else known
    slots = {fx["id"]: {fx["home"], fx["away"]} for fx in resolved_fixtures(bracket, basis)}
    errors = []
    for slot, winner in overrides.items():
        if slot not in slots:
            errors.append(f"{slot} is not pinnable yet")
        elif slot in known and not counterfactual:
            errors.append(f"{slot} already decided ({known[slot]}) - enable counterfactual mode to rewrite")
        elif winner not in slots[slot]:
            errors.append(f"{slot} winner must be one of {sorted(slots[slot])}")
    return errors


JOB = {"phase": "idle", "detail": "", "started": None, "error": None}
JOB_LOCK = threading.Lock()


def _run_refresh_job():
    try:
        JOB.update(phase="refreshing", detail="scrape + refit + simulate", error=None)
        refresh()
        if os.environ.get("THE_ODDS_API_KEY"):
            try:
                import market_anchor
                market_anchor.fetch(os.environ["THE_ODDS_API_KEY"])
            except Exception as e:  # anchor is a benchmark; never block the pipeline
                print(f"market anchor fetch failed: {e}")
        if STATE["samples"] is None:  # stale/absent -> regenerate so the ensemble survives new data
            from uncertainty import bootstrap_samples
            JOB.update(phase="bootstrapping",
                       detail="refitting 16 bootstrap resamples for the uncertainty ensemble")
            df = load_matches(CFG["years"])
            bracket = json.loads((ROOT / "bracket_2026.json").read_text())
            alive = sorted({t for fx in bracket["r16"] for t in (fx["home"], fx["away"])})
            samples = bootstrap_samples(df, 16, CFG["half_life"], CFG["friendly_weight"],
                                        required=alive)
            SAMPLES_FILE.parent.mkdir(exist_ok=True)
            SAMPLES_FILE.write_text(json.dumps({
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "data_max_date": str(df["date"].max().date()),
                "boots": 16, "half_life": CFG["half_life"],
                "friendly_weight": CFG["friendly_weight"], "samples": samples}))
            JOB.update(phase="refreshing", detail="re-simulating with the fresh ensemble")
            refresh()
        JOB.update(phase="idle", detail="")
    except Exception as e:  # surfaced via /api/status; stale payload keeps serving
        JOB.update(phase="error", detail="", error=str(e))


def _start_refresh_job():
    with JOB_LOCK:
        if JOB["phase"] not in ("idle", "error"):
            return False
        JOB.update(phase="starting", detail="", error=None,
                   started=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    threading.Thread(target=_run_refresh_job, daemon=True).start()
    return True


@app.post("/api/refresh")
def api_refresh():
    _apply_knobs(request.get_json(force=True, silent=True) or {})
    started = _start_refresh_job()
    return jsonify({"started": started, "job": JOB}), 202 if started else 409


@app.get("/api/status")
def api_status():
    meta = (STATE["payload"] or {}).get("meta", {})
    return jsonify({"job": JOB, "generated": meta.get("generated"),
                    "uncertainty": meta.get("uncertainty")})


@app.post("/api/whatif")
def api_whatif():
    """Re-run the bracket Monte Carlo with user-pinned winners layered on real results."""
    body = request.get_json(force=True, silent=True) or {}
    overrides = {k: v for k, v in body.get("overrides", {}).items() if v}
    atk, dfn, hfa, rho = STATE["params"]
    bad = [v for v in overrides.values() if v not in atk]
    if bad:
        return jsonify({"error": f"unknown teams: {bad}"}), 400
    bracket = json.loads((ROOT / "bracket_2026.json").read_text())
    counterfactual = bool(body.get("counterfactual"))
    errors = _validate_overrides(bracket, overrides, STATE["payload"]["known"], counterfactual)
    if errors:
        return jsonify({"error": "; ".join(errors)}), 400
    known = _known_with_overrides(bracket, STATE["payload"]["known"], overrides, counterfactual)
    sims = int(_clamp(int(body.get("sims", CFG["sims"])), *KNOB_RANGES["sims"]))
    sampler = body.get("sampler", CFG["sampler"])
    if sampler not in SAMPLERS:
        return jsonify({"error": f"sampler must be one of {SAMPLERS}"}), 400
    if STATE["samples"]:
        probs, paths = run_ensemble(STATE["samples"]["samples"], STATE["pens"],
                                    bracket, known, sims, sampler, goal_scale=CFG["goal_scale"],
                                    scoreline_dispersion=CFG["scoreline_dispersion"],
                                    external_strength=STATE.get("external_strength"),
                                    external_weight=CFG["external_weight"],
                                    form_strength=STATE.get("form_strength"),
                                    form_weight=CFG["form_weight"],
                                    live_strength=STATE.get("live_strength"),
                                    live_weight=CFG["live_weight"])
    else:
        sim = Simulator(atk, dfn, hfa, rho, np.random.default_rng(), pens=STATE["pens"],
                        goal_scale=CFG["goal_scale"],
                        scoreline_dispersion=CFG["scoreline_dispersion"],
                        external_strength=STATE.get("external_strength"),
                        external_weight=CFG["external_weight"],
                        form_strength=STATE.get("form_strength"),
                        form_weight=CFG["form_weight"],
                        live_strength=STATE.get("live_strength"),
                        live_weight=CFG["live_weight"])
        probs, paths = run_tournament(sim, bracket, known, sims, sampler, return_paths=True)
    verdict_sim = Simulator(atk, dfn, hfa, rho, np.random.default_rng(26), pens=STATE["pens"],
                            goal_scale=CFG["goal_scale"],
                            scoreline_dispersion=CFG["scoreline_dispersion"],
                            external_strength=STATE.get("external_strength"),
                            external_weight=CFG["external_weight"],
                            form_strength=STATE.get("form_strength"),
                            form_weight=CFG["form_weight"],
                            live_strength=STATE.get("live_strength"),
                            live_weight=CFG["live_weight"])
    bracket_rows = _bracket_rows(probs)
    verdict = verdict_bracket(verdict_sim, bracket, known)
    return jsonify({"overrides": overrides, "counterfactual": counterfactual,
                    "known": known, "sims": sims, "sampler": sampler,
                    "verdict": verdict,
                    "prediction": _prediction_summary(verdict, bracket_rows),
                    "consensus": build_consensus(paths, bracket, known),
                    "bracket": bracket_rows})


@app.get("/api/market")
def api_market():
    """Market anchor comparison: model vs de-vigged bookmaker champion odds + log-pool blend."""
    if not MARKET_FILE.exists():
        return jsonify({"present": False,
                        "note": "no market snapshot - set THE_ODDS_API_KEY and run market_anchor.py"})
    from market_anchor import log_pool
    market = json.loads(MARKET_FILE.read_text())
    model = {r["team"]: r["champion"] for r in (STATE["payload"] or {}).get("bracket", [])}
    mk = market.get("champion_probs", {})
    blend = log_pool(model, mk, 0.5)
    rows = [{"team": t, "model": model.get(t), "market": mk.get(t),
             "blend": round(blend[t], 5) if t in blend else None,
             "edge": round(model[t] - mk[t], 4) if t in model and t in mk else None}
            for t in sorted(set(model) | set(mk),
                            key=lambda t: -(model.get(t) or mk.get(t) or 0))]
    return jsonify({"present": True, "generated": market.get("generated"),
                    "books": market.get("books"), "rows": rows})


@app.get("/api/consensus")
def api_consensus():
    c = (STATE["payload"] or {}).get("consensus")
    if not c:
        return jsonify({"error": "no consensus yet - refresh first"}), 404
    return jsonify(c)


@app.get("/api/backtest")
def api_backtest():
    if not BACKTEST_FILE.exists():
        return jsonify({"error": "no backtest yet - run backtest.py"}), 404
    return jsonify(json.loads(BACKTEST_FILE.read_text()))


@app.post("/api/backtest")
def api_run_backtest():
    body = request.get_json(force=True, silent=True) or {}
    try:
        half_life = _clamp(float(body.get("half_life", CFG["half_life"])), *KNOB_RANGES["half_life"])
        friendly_weight = _clamp(float(body.get("friendly_weight", CFG["friendly_weight"])), *KNOB_RANGES["friendly_weight"])
        goal_scale = _clamp(float(body.get("goal_scale", CFG["goal_scale"])), *KNOB_RANGES["goal_scale"])
        scoreline_dispersion = _clamp(float(body.get("scoreline_dispersion", CFG["scoreline_dispersion"])),
                                      *KNOB_RANGES["scoreline_dispersion"])
        external_weight = _clamp(float(body.get("external_weight", CFG["external_weight"])), *KNOB_RANGES["external_weight"])
        form_weight = _clamp(float(body.get("form_weight", CFG["form_weight"])), *KNOB_RANGES["form_weight"])
        refit_days = _clamp(int(body.get("refit_days", 45)), 7, 90)
        train_years = _clamp(float(body.get("train_years", CFG["years"])), 2.0, 6.0)
    except (TypeError, ValueError):
        return jsonify({"error": "invalid backtest parameters"}), 400
    with BACKTEST_LOCK:
        return jsonify(write_backtest(
            start=body.get("start", "2026-01-01"),
            refit_days=refit_days,
            train_years=train_years,
            half_life=half_life,
            friendly_weight=friendly_weight,
            goal_scale=goal_scale,
            scoreline_dispersion=scoreline_dispersion,
            external_weight=external_weight,
            form_weight=form_weight,
            verbose=False,
        ))


def auto_refresh_loop(hours):
    while True:
        time.sleep(hours * 3600)
        if not _start_refresh_job():
            print("auto-refresh skipped: a refresh job is already running")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8026)
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    ap.add_argument("--sampler", choices=SAMPLERS, default="antithetic")
    ap.add_argument("--external-weight", type=float, default=DEFAULT_EXTERNAL_WEIGHT)
    ap.add_argument("--scoreline-dispersion", type=float, default=DEFAULT_SCORELINE_DISPERSION)
    ap.add_argument("--form-weight", type=float, default=DEFAULT_FORM_WEIGHT)
    ap.add_argument("--live-weight", type=float, default=DEFAULT_LIVE_WEIGHT)
    ap.add_argument("--auto-refresh-hours", type=float, default=0.25)
    args = ap.parse_args()
    CFG["sims"] = args.sims
    CFG["sampler"] = args.sampler
    CFG["scoreline_dispersion"] = _clamp(args.scoreline_dispersion, *KNOB_RANGES["scoreline_dispersion"])
    CFG["external_weight"] = _clamp(args.external_weight, *KNOB_RANGES["external_weight"])
    CFG["form_weight"] = _clamp(args.form_weight, *KNOB_RANGES["form_weight"])
    CFG["live_weight"] = _clamp(args.live_weight, *KNOB_RANGES["live_weight"])

    loaded = load_state()
    meta = STATE["payload"]["meta"] if loaded else {}
    if (not loaded or meta.get("sims") != CFG["sims"] or meta.get("sampler") != CFG["sampler"]
            or meta.get("half_life_days") != CFG["half_life"]
            or meta.get("friendly_weight") != CFG["friendly_weight"]
            or meta.get("goal_scale") != CFG["goal_scale"]
            or meta.get("scoreline_dispersion") != CFG["scoreline_dispersion"]
            or meta.get("external_weight") != CFG["external_weight"]
            or meta.get("form_weight") != CFG["form_weight"]
            or meta.get("live_weight") != CFG["live_weight"]
            or _state_needs_refresh(meta, max_age_hours=args.auto_refresh_hours)):
        print("refreshing state (scrape + fit + simulate)...")
        refresh()
    print(f"model ready: {STATE['payload']['meta']}")
    if args.auto_refresh_hours > 0:
        threading.Thread(target=auto_refresh_loop, args=(args.auto_refresh_hours,), daemon=True).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
