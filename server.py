"""Flask app: web UI + prediction API + refresh pipeline for the WC-2026 sim.

Run: .venv/Scripts/python server.py [--port 8026] [--sims 10000] [--auto-refresh-hours 6]

Endpoints:
  GET  /               web/index.html
  GET  /api/data       full payload (meta, fixtures+cards, bracket probabilities, ratings)
  GET  /api/predict    ?home=X&away=Y[&venue=C]  Dixon-Coles card for any matchup
  GET  /api/sample     same args; sample one scoreline (pens flag on draws)
  POST /api/refresh    scrape latest results -> refit -> re-simulate
"""
import argparse
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

import fetch_data
from wc_sim import (Simulator, dc_grid, fit_model, known_winners, load_matches,
                    markets, match_rates, run_tournament, shootout_rates, team_params)

ROOT = Path(__file__).parent
STATE_FILE = ROOT / "output" / "state.json"
BACKTEST_FILE = ROOT / "output" / "backtest.json"
app = Flask(__name__, static_folder="web", static_url_path="")
STATE = {"payload": None, "params": None, "pens": {}}  # params = (atk, dfn, hfa, rho)
LOCK = threading.Lock()
CFG = {"sims": 10_000, "half_life": 550.0, "friendly_weight": 1.0, "years": 4.0}
KNOB_RANGES = {"half_life": (100, 2000), "friendly_weight": (0.0, 1.0), "sims": (1000, 50000)}


def card(home, away, venue=""):
    atk, dfn, hfa, rho = STATE["params"]
    lam, mu = match_rates(atk, dfn, hfa, home, away, venue)
    g = dc_grid(lam, mu, rho)
    h, d, a, o25, top = markets(g)
    return {"home": home, "away": away, "venue": venue or "neutral",
            "lam": round(lam, 3), "mu": round(mu, 3),
            "p_home": round(h, 4), "p_draw": round(d, 4), "p_away": round(a, 4),
            "over25": round(o25, 4), "top": [{"score": s, "p": round(p, 4)} for p, s in top],
            "grid": [[round(v, 5) for v in row] for row in g]}


def refresh():
    """Scrape -> refit -> re-simulate -> rebuild payload. Serialized by LOCK."""
    with LOCK:
        fetch_meta = fetch_data.fetch(quiet=True)
        df = load_matches(CFG["years"])
        atk, dfn, hfa, rho = team_params(fit_model(df, CFG["half_life"], CFG["friendly_weight"]))
        STATE["params"] = (atk, dfn, hfa, rho)

        bracket = json.loads((ROOT / "bracket_2026.json").read_text())
        shootouts = pd.read_csv(ROOT / "data" / "shootouts.csv", parse_dates=["date"])
        STATE["pens"] = shootout_rates(shootouts)
        known = known_winners(bracket, df, shootouts)
        sim = Simulator(atk, dfn, hfa, rho, np.random.default_rng(), pens=STATE["pens"])
        probs = run_tournament(sim, bracket, known, CFG["sims"])

        fixtures = []
        for fx in bracket["r16"]:
            c = card(fx["home"], fx["away"], fx["venue_country"])
            c.update(id=fx["id"], date=fx["date"], played=fx["id"] in known,
                     winner=known.get(fx["id"]))
            fixtures.append(c)

        STATE["payload"] = {
            "meta": {**fetch_meta, "trained_matches": len(df),
                     "train_from": str(df["date"].min().date()),
                     "teams": df["home_team"].nunique(),
                     "hfa": round(hfa, 3), "rho": round(rho, 3),
                     "sims": CFG["sims"], "half_life_days": CFG["half_life"],
                     "friendly_weight": CFG["friendly_weight"],
                     "generated": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            "fixtures": fixtures,
            "tree": {k: bracket[k] for k in ("qf", "sf", "final")},
            "known": known,
            "bracket": sorted(
                ({"team": t, "qf": round(p[0], 4), "sf": round(p[1], 4),
                  "final": round(p[2], 4), "champion": round(p[3], 4)}
                 for t, p in probs.items()), key=lambda r: -r["champion"]),
            "teams": sorted(atk),
            "ratings": sorted(
                ({"team": t, "attack": round(atk[t], 3), "defence": round(dfn[t], 3)}
                 for t in atk), key=lambda r: -(r["attack"] - r["defence"]))[:30],
        }
        STATE_FILE.parent.mkdir(exist_ok=True)
        STATE_FILE.write_text(json.dumps(
            {"payload": STATE["payload"], "pens": STATE["pens"],
             "params": {"attack": atk, "defence": dfn, "hfa": hfa, "rho": rho}}))
    return STATE["payload"]["meta"]


def load_state():
    if STATE_FILE.exists():
        s = json.loads(STATE_FILE.read_text())
        STATE["payload"] = s["payload"]
        STATE["pens"] = s.get("pens", {})
        p = s["params"]
        STATE["params"] = (p["attack"], p["defence"], p["hfa"], p["rho"])
        return True
    return False


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/data")
def api_data():
    return jsonify(STATE["payload"])


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
    atk, dfn, hfa, rho = STATE["params"]
    sim = Simulator(atk, dfn, hfa, rho, np.random.default_rng(), pens=STATE["pens"])
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
    for k in ("half_life", "friendly_weight", "sims"):
        if k in body:
            lo, hi = KNOB_RANGES[k]
            v = min(max(float(body[k]), lo), hi)
            CFG[k] = int(v) if k == "sims" else v
            changed[k] = CFG[k]
    return changed


@app.post("/api/refresh")
def api_refresh():
    _apply_knobs(request.get_json(force=True, silent=True) or {})
    return jsonify(refresh())


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
    known = {**STATE["payload"]["known"], **overrides}
    sims = min(int(body.get("sims", 5000)), 20000)
    sim = Simulator(atk, dfn, hfa, rho, np.random.default_rng(), pens=STATE["pens"])
    probs = run_tournament(sim, bracket, known, sims)
    return jsonify({"overrides": overrides, "sims": sims, "bracket": sorted(
        ({"team": t, "qf": round(p[0], 4), "sf": round(p[1], 4),
          "final": round(p[2], 4), "champion": round(p[3], 4)}
         for t, p in probs.items()), key=lambda r: -r["champion"])})


@app.get("/api/backtest")
def api_backtest():
    if not BACKTEST_FILE.exists():
        return jsonify({"error": "no backtest yet - run backtest.py"}), 404
    return jsonify(json.loads(BACKTEST_FILE.read_text()))


def auto_refresh_loop(hours):
    while True:
        time.sleep(hours * 3600)
        try:
            refresh()
        except Exception as e:  # keep serving stale data if a scheduled scrape fails
            print(f"auto-refresh failed: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8026)
    ap.add_argument("--sims", type=int, default=10_000)
    ap.add_argument("--auto-refresh-hours", type=float, default=6.0)
    args = ap.parse_args()
    CFG["sims"] = args.sims

    if not load_state():
        print("no saved state - running first refresh (scrape + fit + simulate)...")
        refresh()
    print(f"model ready: {STATE['payload']['meta']}")
    if args.auto_refresh_hours > 0:
        threading.Thread(target=auto_refresh_loop, args=(args.auto_refresh_hours,), daemon=True).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
