"""Sync live FIFA men's rankings into data/fifa_rankings_latest.csv."""
from __future__ import annotations

import argparse
import csv
import json
import ssl
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

from wc_sim import ROOT

OUT = ROOT / "data" / "fifa_rankings_latest.csv"
SOURCE = "https://www.fifa.com/en/world-rankings"
API = "https://api.fifa.com/api/v3/fifarankings/rankings/live?gender=1&count=300&language=en&sportType=0"

TEAM_ALIASES = {
    "Cabo Verde": "Cape Verde",
    "China PR": "China",
    "Czechia": "Czech Republic",
    "DPR Korea": "North Korea",
    "IR Iran": "Iran",
    "Ireland": "Republic of Ireland",
    "Korea, North": "North Korea",
    "Korea, South": "South Korea",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "USA": "United States",
}


def canonical_team_name(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            return json.load(r)
    except urllib.error.URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise
        # ponytail: public FIFA ranking feed; remove fallback once local cert store trusts FIFA's chain.
        with urllib.request.urlopen(req, timeout=45, context=ssl._create_unverified_context()) as r:
            return json.load(r)


def _name(row: dict) -> str:
    names = row.get("TeamName") or []
    for item in names:
        if item.get("Locale", "").lower().startswith("en"):
            return item["Description"]
    return names[0]["Description"]


def rows_from_payload(payload: dict, source_date: str | None = None) -> list[dict]:
    rows = []
    for row in payload.get("Results", []):
        team = canonical_team_name(_name(row))
        rows.append({
            "team": team,
            "fifa_ranking": int(row["Rank"]),
            "source_date": source_date or date.today().isoformat(),
            "source": SOURCE,
            "country_code": row.get("IdCountry"),
            "total_points": round(float(row.get("TotalPoints", 0.0)), 3),
            "previous_rank": row.get("PrevRank"),
            "ranking_movement": row.get("RankingMovement"),
            "confederation": row.get("ConfederationName"),
        })
    rows.sort(key=lambda r: r["fifa_ranking"])
    return rows


def sync_rankings(out: Path = OUT) -> list[dict]:
    rows = rows_from_payload(_fetch_json(API))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rows = sync_rankings(args.out)
    print(f"wrote {len(rows)} live FIFA rankings to {args.out}")
    print(", ".join(f"{r['fifa_ranking']}. {r['team']}" for r in rows[:5]))


if __name__ == "__main__":
    main()
