"""Confirmed-lineup ingestion and conservative missing-core adjustments."""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from match_features import COMPLETE, ESPN_SCOREBOARD, ESPN_SUMMARY, TEAM_ALIASES

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "player_match_features.csv"
AVAILABILITY_FILE = ROOT / "data" / "lineup_availability.json"
META_FILE = ROOT / "data" / "lineup_availability_meta.json"
PLAYER_POOL = ROOT / "output" / "external" / "player_pool.csv"
TEAM_MART = ROOT / "output" / "external" / "project_team_enrichment.csv"


def _fetch_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _key(name):
    text = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _team(name):
    return TEAM_ALIASES.get(name, name)


def _stats(player):
    return {stat.get("name"): stat.get("value") for stat in player.get("stats", [])}


def player_rows_from_summary(event, summary):
    status = event.get("status", {}).get("type", {}).get("name", "")
    date = str(event.get("date", ""))[:10]
    event_id = str(event.get("id", ""))
    teams = [_team(r.get("team", {}).get("displayName", "")) for r in summary.get("rosters", [])]
    rows = []
    for roster in summary.get("rosters", []):
        team = _team(roster.get("team", {}).get("displayName", ""))
        opponent = next((name for name in teams if name and name != team), "")
        starters = sum(bool(player.get("starter")) for player in roster.get("roster", []))
        for player in roster.get("roster", []):
            athlete = player.get("athlete", {})
            if not athlete.get("displayName"):
                continue
            stats = _stats(player)
            rows.append({
                "date": date,
                "espn_event_id": event_id,
                "event_status": status,
                "team": team,
                "opponent": opponent,
                "home_away": roster.get("homeAway"),
                "formation": roster.get("formation"),
                "lineup_confirmed": starters >= 11,
                "player_id": athlete.get("id"),
                "player": athlete.get("displayName"),
                "starter": bool(player.get("starter")),
                "appearance": float(stats.get("appearances") or 0) > 0,
                "position": player.get("position", {}).get("displayName"),
                "formation_place": player.get("formationPlace"),
                "goals": float(stats.get("totalGoals") or 0),
                "assists": float(stats.get("goalAssists") or 0),
                "shots": float(stats.get("totalShots") or 0),
                "shots_on_target": float(stats.get("shotsOnTarget") or 0),
                "saves": float(stats.get("saves") or 0),
            })
    return rows


def load_player_match_features(path=DATA_FILE):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"espn_event_id": str}, parse_dates=["date"])


def build_confirmed_lineup_availability(rows, player_pool, team_mart, window=5):
    if rows is None or rows.empty or player_pool is None or player_pool.empty:
        return {}, {"present": False, "reason": "no player lineup history"}
    df = rows.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["player_key"] = df["player"].map(_key)
    pool = player_pool.copy()
    pool["player_key"] = pool["player"].map(_key)
    pool["market_value_in_eur"] = pd.to_numeric(pool["market_value_in_eur"], errors="coerce")
    values = {(row.team, row.player_key): float(row.market_value_in_eur)
              for row in pool.dropna(subset=["team", "market_value_in_eur"]).itertuples()}
    roles = {(row.team, row.player_key): row.position for row in pool.itertuples()}
    totals = {}
    if team_mart is not None and not team_mart.empty:
        totals = dict(zip(team_mart["team"], pd.to_numeric(
            team_mart["top23_market_value"], errors="coerce")))

    availability, details = {}, []
    scheduled = df[(df["event_status"] == "STATUS_SCHEDULED") & df["lineup_confirmed"].astype(bool)]
    for (event_id, team), confirmed in scheduled.groupby(["espn_event_id", "team"]):
        actual = confirmed[confirmed["starter"].astype(bool)]
        if len(actual) < 11:
            continue
        day = confirmed["date"].max()
        history = df[(df["team"] == team) & (df["event_status"].isin(COMPLETE))
                     & (df["date"] < day) & df["starter"].astype(bool)]
        recent_events = (history[["date", "espn_event_id"]].drop_duplicates()
                         .sort_values(["date", "espn_event_id"]).tail(window)["espn_event_id"])
        history = history[history["espn_event_id"].isin(recent_events)]
        if history.empty:
            continue
        core = (history.groupby(["player_key", "player"], as_index=False)
                .agg(starts=("espn_event_id", "nunique"), latest=("date", "max"))
                .sort_values(["starts", "latest"], ascending=False).head(11))
        actual_keys = set(actual["player_key"])
        expected_values = [values.get((team, key)) for key in core["player_key"]]
        actual_values = [values.get((team, key)) for key in actual["player_key"]]
        expected_known = [value for value in expected_values if value is not None and math.isfinite(value)]
        actual_known = [value for value in actual_values if value is not None and math.isfinite(value)]
        top23 = totals.get(team)
        if len(expected_known) < 8 or len(actual_known) < 8 or not top23 or not math.isfinite(top23):
            continue
        missing = core[~core["player_key"].isin(actual_keys)]
        missing_values = [(row.player, row.player_key, values.get((team, row.player_key), 0.0))
                          for row in missing.itertuples()]
        missing_total = sum(max(0.0, value) for _, _, value in missing_values)
        shortfall = max(0.0, sum(expected_known) - sum(actual_known))
        value_share = min(0.5, shortfall / float(top23))
        if value_share <= 0 or missing_total <= 0:
            continue
        entries = [{
            "player": player,
            "value_share": round(value_share * value / missing_total, 6),
            "role": roles.get((team, key)),
            "note": "confirmed XI value shortfall versus recent starting core",
            "source": "ESPN confirmed lineup",
        } for player, key, value in missing_values if value > 0]
        if entries:
            availability[team] = entries
            details.append({"event_id": event_id, "team": team, "date": day.date().isoformat(),
                            "missing": [entry["player"] for entry in entries],
                            "missing_value_share": round(value_share, 4),
                            "expected_value_coverage": len(expected_known),
                            "confirmed_value_coverage": len(actual_known)})
    return availability, {"present": bool(availability), "applied": details,
                          "policy": "confirmed scheduled XI downside only", "window": window}


def fetch_lineup_signals(start="2026-06-01", end=None, quiet=False):
    end = end or (datetime.now(timezone.utc).date() + timedelta(days=2)).isoformat()
    span = f"{pd.Timestamp(start):%Y%m%d}-{pd.Timestamp(end):%Y%m%d}"
    events = _fetch_json(ESPN_SCOREBOARD.format(span=span)).get("events", [])
    existing = load_player_match_features()
    stable_ids = set()
    if not existing.empty:
        stable = existing[existing["event_status"].isin(COMPLETE)]
        counts = stable[stable["starter"].astype(bool)].groupby("espn_event_id")["team"].count()
        stable_ids = set(counts[counts >= 22].index.astype(str))
    need = [event for event in events
            if str(event.get("id")) not in stable_ids
            or event.get("status", {}).get("type", {}).get("name") not in COMPLETE]
    summaries = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_json, ESPN_SUMMARY.format(event_id=event["id"])): event
                   for event in need}
        for future, event in futures.items():
            try:
                summary = future.result()
                if summary.get("rosters"):
                    summaries[str(event["id"])] = (event, summary)
            except Exception:
                continue
    replace_ids = set(summaries)
    frames = []
    if not existing.empty:
        frames.append(existing[~existing["espn_event_id"].astype(str).isin(replace_ids)])
    fresh = [row for event, summary in summaries.values()
             for row in player_rows_from_summary(event, summary)]
    if fresh:
        frames.append(pd.DataFrame(fresh))
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not rows.empty:
        rows = rows.sort_values(["date", "espn_event_id", "team", "starter", "player"],
                                ascending=[True, True, True, False, True])
    DATA_FILE.parent.mkdir(exist_ok=True)
    rows.to_csv(DATA_FILE, index=False)

    player_pool = pd.read_csv(PLAYER_POOL) if PLAYER_POOL.exists() else pd.DataFrame()
    team_mart = pd.read_csv(TEAM_MART) if TEAM_MART.exists() else pd.DataFrame()
    availability, availability_meta = build_confirmed_lineup_availability(
        rows, player_pool, team_mart)
    AVAILABILITY_FILE.write_text(json.dumps(availability, indent=1, ensure_ascii=False))
    meta = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": len(rows), "events": int(rows["espn_event_id"].nunique()) if not rows.empty else 0,
            "confirmed_upcoming_teams": int(rows[(rows.get("event_status") == "STATUS_SCHEDULED")
                                                  & rows.get("lineup_confirmed", False)]["team"].nunique())
            if not rows.empty else 0,
            "availability": availability_meta}
    META_FILE.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    if not quiet:
        print(f"player lineups: {meta['rows']} rows, {meta['events']} events, "
              f"{meta['confirmed_upcoming_teams']} confirmed upcoming teams")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2026-06-01")
    parser.add_argument("--end")
    args = parser.parse_args()
    fetch_lineup_signals(args.start, args.end)


if __name__ == "__main__":
    main()
