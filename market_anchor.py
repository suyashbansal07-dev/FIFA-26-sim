"""Optional market-odds anchor (benchmark, not a model input).

Fetches outright World Cup winner odds from the-odds-api.com (free key,
THE_ODDS_API_KEY env var), de-vigs each bookmaker, averages implied
probabilities across books, and stores a comparison against the model's
championship odds. Deliberately kept OUT of the fit: the market is the
benchmark to beat, not a feature (see docs/EVIDENCE_LOG.md).

Run: .venv/Scripts/python market_anchor.py
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
OUT_FILE = ROOT / "output" / "market_odds.json"
API = ("https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup_winner/odds/"
       "?apiKey={key}&regions=eu,uk,us&markets=outrights&oddsFormat=decimal")
# bookmaker naming -> dataset naming
TEAM_ALIASES = {"USA": "United States", "Bosnia and Herzegovina": "Bosnia and Herzegovina",
                "Korea Republic": "South Korea", "Ivory Coast": "Ivory Coast"}


def devig(decimal_odds: dict) -> dict:
    """Decimal odds -> implied probabilities with the bookmaker margin removed."""
    implied = {t: 1.0 / o for t, o in decimal_odds.items() if o and o > 1.0}
    total = sum(implied.values())
    return {t: p / total for t, p in implied.items()} if total > 0 else {}


def log_pool(p_model: dict, p_market: dict, w: float = 0.5) -> dict:
    """Logarithmic opinion pool over the teams both sides price."""
    common = {t for t in p_model if p_model[t] > 0} & {t for t in p_market if p_market[t] > 0}
    if not common:
        return {}
    raw = {t: p_model[t] ** (1 - w) * p_market[t] ** w for t in common}
    total = sum(raw.values())
    return {t: v / total for t, v in raw.items()}


def market_probs_from_events(events) -> tuple[dict, int]:
    """Average de-vigged winner probabilities across all bookmakers in the feed."""
    sums, counts, books = {}, {}, 0
    for ev in events:
        for bm in ev.get("bookmakers", []):
            for mk in bm.get("markets", []):
                if mk.get("key") != "outrights":
                    continue
                odds = {TEAM_ALIASES.get(o["name"], o["name"]): o.get("price")
                        for o in mk.get("outcomes", [])}
                probs = devig(odds)
                if not probs:
                    continue
                books += 1
                for t, p in probs.items():
                    sums[t] = sums.get(t, 0.0) + p
                    counts[t] = counts.get(t, 0) + 1
    return {t: sums[t] / counts[t] for t in sums}, books


def fetch(key: str) -> dict:
    with urllib.request.urlopen(API.format(key=key), timeout=30) as r:
        events = json.load(r)
    probs, books = market_probs_from_events(events)
    report = {"present": bool(probs), "source": "the-odds-api.com", "books": books,
              "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "champion_probs": {t: round(p, 5) for t, p in
                                 sorted(probs.items(), key=lambda kv: -kv[1])}}
    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, indent=1))
    return report


def main():
    key = os.environ.get("THE_ODDS_API_KEY")
    if not key:
        print("THE_ODDS_API_KEY not set - get a free key at the-odds-api.com; "
              "the server will report the anchor as unavailable until then.")
        return
    r = fetch(key)
    print(f"market anchor: {r['books']} bookmaker books averaged")
    for t, p in list(r["champion_probs"].items())[:8]:
        print(f"  {t:15} {p:.1%}")
    print(f"wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
