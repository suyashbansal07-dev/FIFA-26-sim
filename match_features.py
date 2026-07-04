"""Optional match-wise xG/stat feature ingestion.

ESPN's WC API carries completed-match team stats in the scoreboard payload and
xG-ish team totals via summary leader xGC. These are post-match observations:
diagnostics may use them, forecasts do not.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={span}"
ESPN_SUMMARY = "https://site.web.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event={event_id}"
COMPLETE = {"STATUS_FULL_TIME", "STATUS_FINAL_AET", "STATUS_FINAL_PEN"}
TEAM_ALIASES = {"Congo DR": "DR Congo", "Bosnia-Herzegovina": "Bosnia and Herzegovina", "USA": "United States"}
COUNTRY_ALIASES = {"USA": "United States"}
STAT_MAP = {
    "totalShots": "shots",
    "shotsOnTarget": "sot",
    "wonCorners": "corners",
    "possessionPct": "possession",
    "foulsCommitted": "fouls",
    "accuratePasses": "accurate_passes",
    "totalPasses": "passes",
    "passPct": "pass_pct",
}
FEATURE_COLS = ["xg", *STAT_MAP.values()]


def _team(name):
    return TEAM_ALIASES.get(name, name)


def _num(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace("%", "").replace(",", "").strip())
    except ValueError:
        return None


def _fetch_json(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def _event_id(value):
    if pd.isna(value):
        return ""
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value)


def _stats(comp):
    return {s.get("name"): _num(s.get("value", s.get("displayValue"))) for s in comp.get("statistics", [])}


def _summary_xg_by_team(summary):
    """Return team xG inferred from opponent goalkeeper xGC, when ESPN exposes it."""
    xgc = {}
    for group in summary.get("leaders", []):
        team = _team(group.get("team", {}).get("displayName", ""))
        if not team:
            continue
        for category in group.get("leaders", []):
            if category.get("name") != "saves":
                continue
            for leader in category.get("leaders", []):
                for stat in leader.get("statistics", []):
                    if stat.get("name") == "expectedGoalsConceded":
                        xgc[team] = _num(stat.get("value", stat.get("displayValue")))
    if len(xgc) != 2:
        return {}
    a, b = list(xgc)
    return {a: xgc.get(b), b: xgc.get(a)}


def feature_row_from_event(event, summary=None):
    if event.get("status", {}).get("type", {}).get("name") not in COMPLETE:
        return None
    comp = event.get("competitions", [{}])[0]
    sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
    if "home" not in sides or "away" not in sides:
        return None
    home, away = _team(sides["home"]["team"]["displayName"]), _team(sides["away"]["team"]["displayName"])
    addr = comp.get("venue", {}).get("address", {})
    country = COUNTRY_ALIASES.get(addr.get("country", ""), addr.get("country", ""))
    row = {
        "date": event["date"][:10],
        "espn_event_id": event.get("id"),
        "home_team": home,
        "away_team": away,
        "venue_country": country,
        "neutral": home != country,
    }
    xg = _summary_xg_by_team(summary or {})
    for side, team in (("home", home), ("away", away)):
        stats = _stats(sides[side])
        row[f"{side}_xg"] = xg.get(team)
        for src, dst in STAT_MAP.items():
            row[f"{side}_{dst}"] = stats.get(src)
    return row


def fetch_espn_match_features(start="2026-06-01", end=None, include_xg=True, quiet=False):
    end = end or datetime.now(timezone.utc).date().isoformat()
    span = f"{pd.Timestamp(start):%Y%m%d}-{pd.Timestamp(end):%Y%m%d}"
    events = _fetch_json(ESPN_SCOREBOARD.format(span=span)).get("events", [])
    completed = [e for e in events if e.get("status", {}).get("type", {}).get("name") in COMPLETE]
    existing = load_match_features()
    existing_by_id = {}
    if not existing.empty:
        existing_by_id = {_event_id(r["espn_event_id"]): r for r in existing.to_dict("records")}
    summaries = {}
    if include_xg:
        need_summary = [
            e for e in completed
            if not existing_by_id.get(_event_id(e.get("id")))
            or pd.isna(existing_by_id[_event_id(e.get("id"))].get("home_xg"))
            or pd.isna(existing_by_id[_event_id(e.get("id"))].get("away_xg"))
        ]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch_json, ESPN_SUMMARY.format(event_id=e["id"])): e["id"] for e in need_summary}
            for fut, event_id in futures.items():
                try:
                    summaries[event_id] = fut.result()
                except Exception:
                    summaries[event_id] = {}
    rows = []
    for e in completed:
        row = feature_row_from_event(e, summaries.get(e.get("id")))
        old = existing_by_id.get(_event_id(e.get("id")))
        if row and old:
            for side in ("home", "away"):
                if pd.isna(row.get(f"{side}_xg")):
                    row[f"{side}_xg"] = old.get(f"{side}_xg")
        rows.append(row)
    rows = [r for r in rows if r]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates("espn_event_id").sort_values(["date", "espn_event_id"])
    DATA.mkdir(exist_ok=True)
    df.to_csv(DATA / "match_features.csv", index=False)
    meta = feature_coverage(df)
    if not quiet:
        print(f"match features: {meta['rows']} rows | xG coverage {meta['coverage'].get('xg', 0):.1%}")
    return meta


def load_match_features(root=ROOT):
    path = root / "data" / "match_features.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])


def _key(date, home, away):
    return (pd.Timestamp(date).date().isoformat(), home, away)


def find_match_feature(features, date, home, away):
    if features is None or features.empty:
        return None
    hit = features[(features["date"].dt.date == pd.Timestamp(date).date())
                   & (features["home_team"] == home) & (features["away_team"] == away)]
    swapped = False
    if hit.empty:
        hit = features[(features["date"].dt.date == pd.Timestamp(date).date())
                       & (features["home_team"] == away) & (features["away_team"] == home)]
        swapped = not hit.empty
    if hit.empty:
        return None
    row = hit.iloc[-1].to_dict()
    if not swapped:
        return row
    out = {}
    for k, v in row.items():
        if k.startswith("home_"):
            out["away_" + k[5:]] = v
        elif k.startswith("away_"):
            out["home_" + k[5:]] = v
        else:
            out[k] = v
    return out


def attach_features(records, features):
    for rec in records:
        feat = find_match_feature(features, rec["date"], rec["home_team"], rec["away_team"])
        rec["has_match_features"] = bool(feat)
        if not feat:
            continue
        for side in ("home", "away"):
            for col in FEATURE_COLS:
                rec[f"{side}_{col}"] = feat.get(f"{side}_{col}")
    return records


def feature_coverage(features_or_records):
    if isinstance(features_or_records, pd.DataFrame):
        rows = features_or_records.to_dict("records")
    else:
        rows = list(features_or_records)
    out = {"present": bool(rows), "rows": len(rows), "coverage": {}}
    for col in FEATURE_COLS:
        vals = [r.get(f"home_{col}") for r in rows] + [r.get(f"away_{col}") for r in rows]
        vals = [v for v in vals if pd.notna(v)]
        out["coverage"][col] = round(len(vals) / (2 * len(rows)), 3) if rows else 0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end")
    ap.add_argument("--no-xg", action="store_true")
    args = ap.parse_args()
    fetch_espn_match_features(args.start, args.end, include_xg=not args.no_xg)


if __name__ == "__main__":
    main()
