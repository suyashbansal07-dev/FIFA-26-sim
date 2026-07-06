"""Scrapers: martj42 bulk dataset + ESPN same-day top-up -> data/matches.csv, data/shootouts.csv.

martj42 can lag a day or two mid-tournament; the ESPN scoreboard API fills the gap
so the model retrains on every finished game. Run directly or via server /api/refresh.
"""
import json
import urllib.request
from datetime import timedelta
from pathlib import Path

import pandas as pd

from match_features import fetch_espn_match_features

BASE = "https://raw.githubusercontent.com/martj42/international_results/master/"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={span}"
DATA = Path(__file__).parent / "data"
COMPLETE = {"STATUS_FULL_TIME", "STATUS_FINAL_AET", "STATUS_FINAL_PEN"}
# ESPN display names -> martj42 names (only teams that can still appear at this WC need care)
TEAM_ALIASES = {"Congo DR": "DR Congo", "Bosnia-Herzegovina": "Bosnia and Herzegovina", "USA": "United States"}
COUNTRY_ALIASES = {"USA": "United States"}


def _team(name):
    return TEAM_ALIASES.get(name, name)


def espn_topup(matches, shootouts, events=None, today=None):
    """Append finished WC matches that ESPN has but the bulk dataset doesn't yet."""
    played = matches.dropna(subset=["home_score"])
    newest = played["date"].max()
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    if newest > today and events is None:
        return matches, shootouts, 0

    if events is None:
        span = f"{(newest - timedelta(days=1)):%Y%m%d}-{today:%Y%m%d}"
        with urllib.request.urlopen(ESPN.format(span=span), timeout=30) as r:
            events = json.load(r).get("events", [])

    recent = matches[matches["date"] >= newest - timedelta(days=3)]

    def pair_rows(frame, h, a, date):
        near = frame[(frame["date"] - date).abs() <= pd.Timedelta(days=2)]
        return near[(((near["home_team"] == h) & (near["away_team"] == a))
                     | ((near["home_team"] == a) & (near["away_team"] == h)))]

    new_rows, new_pens = [], []
    consumed = 0
    for ev in events:
        if ev["status"]["type"]["name"] not in COMPLETE:
            continue
        comp = ev["competitions"][0]
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        h, a = _team(sides["home"]["team"]["displayName"]), _team(sides["away"]["team"]["displayName"])
        date = pd.Timestamp(ev["date"][:10])
        if not pair_rows(recent.dropna(subset=["home_score"]), h, a, date).empty:
            continue
        addr = comp.get("venue", {}).get("address", {})
        country = COUNTRY_ALIASES.get(addr.get("country", ""), addr.get("country", ""))
        row = {
            "date": date, "home_team": h, "away_team": a,
            "home_score": int(sides["home"]["score"]), "away_score": int(sides["away"]["score"]),
            "tournament": "FIFA World Cup", "city": addr.get("city", ""), "country": country,
            "neutral": h != country,
        }
        pending = pair_rows(recent[recent["home_score"].isna()], h, a, date)
        if not pending.empty:
            i = pending.index[-1]
            for k, v in row.items():
                matches.loc[i, k] = v
            consumed += 1
        else:
            new_rows.append(row)
        if ev["status"]["type"]["name"] == "STATUS_FINAL_PEN":
            winner = next(_team(c["team"]["displayName"]) for c in comp["competitors"] if c.get("winner"))
            new_pens.append({"date": date, "home_team": h, "away_team": a,
                             "winner": winner, "first_shooter": ""})

    if new_rows:
        matches = pd.concat([matches, pd.DataFrame(new_rows)], ignore_index=True)
    if new_pens:
        shootouts = pd.concat([shootouts, pd.DataFrame(new_pens)], ignore_index=True)
    return matches, shootouts, consumed + len(new_rows)


def fetch(quiet=False):
    """Download + merge both sources. Returns freshness meta dict."""
    DATA.mkdir(exist_ok=True)
    for src, dst in [("results.csv", "matches.csv"), ("shootouts.csv", "shootouts.csv")]:
        urllib.request.urlretrieve(BASE + src, DATA / dst)

    matches = pd.read_csv(DATA / "matches.csv", parse_dates=["date"])
    shootouts = pd.read_csv(DATA / "shootouts.csv", parse_dates=["date"])
    try:
        matches, shootouts, n_new = espn_topup(matches, shootouts)
    except Exception as e:  # scraper failure must not block re-runs on stale-but-usable data
        n_new = 0
        if not quiet:
            print(f"WARNING: ESPN top-up failed ({e}); continuing with bulk dataset only")
    matches.to_csv(DATA / "matches.csv", index=False)
    shootouts.to_csv(DATA / "shootouts.csv", index=False)
    try:
        feature_meta = fetch_espn_match_features(quiet=True)
    except Exception as e:  # optional stats/xG must not block core result ingestion
        feature_meta = {"error": str(e)}
        if not quiet:
            print(f"WARNING: ESPN match-feature fetch failed ({e}); continuing without stats/xG")

    played = matches.dropna(subset=["home_score"])
    wc = played[(played["tournament"] == "FIFA World Cup") & (played["date"] >= "2026-06-01")]
    meta = {"rows": len(matches), "newest_result": str(played["date"].max().date()),
            "wc2026_played": len(wc), "espn_topup_rows": n_new,
            "match_features": feature_meta}
    if not quiet:
        print(f"rows: {meta['rows']} | newest played result: {meta['newest_result']} | "
              f"WC-2026 played: {meta['wc2026_played']} | ESPN top-up: {n_new}")
    return meta


if __name__ == "__main__":
    fetch()
