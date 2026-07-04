"""Build compact player/market enrichment tables from open Transfermarkt data.

Run:
  .venv/Scripts/python external_data.py

Outputs live under output/external/ and are ignored by git. This keeps the repo
small while making the player/market layer reproducible.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from wc_sim import ROOT

OUT = ROOT / "output" / "external"
BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TABLES = {
    "players": f"{BASE}/players.csv.gz",
    "national_teams": f"{BASE}/national_teams.csv.gz",
    "games": f"{BASE}/games.csv.gz",
    "appearances": f"{BASE}/appearances.csv.gz",
    "game_lineups": f"{BASE}/game_lineups.csv.gz",
}


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def build_external_mart(start="2026-06-01", out_dir=OUT, include_usage=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.sql("INSTALL httpfs; LOAD httpfs;")
    except Exception:
        con.sql("LOAD httpfs;")

    con.sql(f"create or replace temp view players as select * from read_csv_auto('{TABLES['players']}')")
    con.sql(f"create or replace temp view national_teams as select * from read_csv_auto('{TABLES['national_teams']}')")
    con.sql("""
        create or replace temp view player_rank as
        select nt.national_team_id, nt.name as team, nt.confederation, nt.fifa_ranking,
               nt.squad_size, nt.average_age, nt.total_market_value,
               p.player_id, p.name as player, p.position, p.sub_position, p.foot,
               p.height_in_cm, p.date_of_birth, p.international_caps, p.international_goals,
               p.current_club_name, p.market_value_in_eur,
               row_number() over (
                   partition by nt.national_team_id
                   order by p.market_value_in_eur desc nulls last, p.international_caps desc nulls last
               ) as market_rank
        from national_teams nt
        left join players p on p.current_national_team_id = nt.national_team_id
        where nt.last_season >= 2025
    """)
    con.sql("""
        create or replace temp view team_strength as
        select team, confederation, fifa_ranking, squad_size, average_age,
               max(total_market_value) as tm_total_market_value,
               count(player_id) as current_nt_players,
               sum(coalesce(market_value_in_eur, 0)) as listed_market_value,
               sum(case when market_rank <= 11 then coalesce(market_value_in_eur, 0) else 0 end) as top11_market_value,
               sum(case when market_rank <= 23 then coalesce(market_value_in_eur, 0) else 0 end) as top23_market_value,
               sum(coalesce(international_caps, 0)) as squad_caps,
               sum(coalesce(international_goals, 0)) as squad_goals
        from player_rank
        group by all
    """)
    if include_usage:
        con.sql(f"create or replace temp view games as select * from read_csv_auto('{TABLES['games']}')")
        con.sql(f"create or replace temp view appearances as select * from read_csv_auto('{TABLES['appearances']}')")
        con.sql(
            "create or replace temp view game_lineups_raw as "
            f"select * from read_csv_auto('{TABLES['game_lineups']}', "
            "strict_mode=false, null_padding=true, all_varchar=true)"
        )
        con.sql("""
            create or replace temp view game_lineups as
            select try_cast(date as date) as date, try_cast(game_id as bigint) as game_id,
                   try_cast(player_id as bigint) as player_id, try_cast(club_id as bigint) as club_id,
                   player_name, type, position,
                   try_cast(team_captain as integer) as team_captain
            from game_lineups_raw
        """)
        con.sql(f"""
            create or replace temp view fiwc_2026_appearances as
            select nt.name as team, count(*) as player_appearances,
                   sum(a.minutes_played) as minutes_played,
                   sum(a.goals) as goals, sum(a.assists) as assists,
                   sum(a.yellow_cards) as yellow_cards, sum(a.red_cards) as red_cards
            from appearances a
            join games g on g.game_id = a.game_id
            left join national_teams nt on nt.national_team_id = a.player_club_id
            where g.competition_id = 'FIWC' and g.date >= '{start}'
            group by nt.name
        """)
        con.sql(f"""
            create or replace temp view fiwc_2026_starts as
            select nt.name as team,
                   sum(case when lower(type) = 'starting_lineup' then 1 else 0 end) as starts,
                   sum(case when team_captain = 1 then 1 else 0 end) as captain_starts
            from game_lineups gl
            join games g on g.game_id = gl.game_id
            left join national_teams nt on nt.national_team_id = gl.club_id
            where g.competition_id = 'FIWC' and g.date >= '{start}'
            group by nt.name
        """)
    else:
        con.sql("""
            create or replace temp view fiwc_2026_appearances as
            select team, 0 as player_appearances, 0 as minutes_played, 0 as goals, 0 as assists,
                   0 as yellow_cards, 0 as red_cards
            from team_strength where false
        """)
        con.sql("""
            create or replace temp view fiwc_2026_starts as
            select team, 0 as starts, 0 as captain_starts
            from team_strength where false
        """)
    con.sql("""
        create or replace temp view project_team_enrichment as
        select ts.*, coalesce(a.player_appearances, 0) as fiwc_player_appearances,
               coalesce(a.minutes_played, 0) as fiwc_minutes,
               coalesce(a.goals, 0) as fiwc_player_goals,
               coalesce(a.assists, 0) as fiwc_assists,
               coalesce(a.yellow_cards, 0) as fiwc_yellow_cards,
               coalesce(a.red_cards, 0) as fiwc_red_cards,
               coalesce(s.starts, 0) as fiwc_starts,
               coalesce(s.captain_starts, 0) as fiwc_captain_starts
        from team_strength ts
        left join fiwc_2026_appearances a using(team)
        left join fiwc_2026_starts s using(team)
    """)
    for view in ("team_strength", "fiwc_2026_appearances", "fiwc_2026_starts", "project_team_enrichment"):
        con.sql(f"copy (select * from {view}) to '{_sql_path(out_dir / (view + '.csv'))}' (header, delimiter ',')")
    sample = con.sql("""
        select team, fifa_ranking, current_nt_players, top11_market_value, top23_market_value,
               fiwc_minutes, fiwc_player_goals
        from project_team_enrichment
        order by top23_market_value desc nulls last
        limit 12
    """).fetchdf()
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "dcaribou/transfermarkt-datasets",
        "source_base": BASE,
        "start": start,
        "include_usage": include_usage,
        "rows": {
            "team_strength": int(con.sql("select count(*) from team_strength").fetchone()[0]),
            "project_team_enrichment": int(con.sql("select count(*) from project_team_enrichment").fetchone()[0]),
            "fiwc_2026_appearances": int(con.sql("select count(*) from fiwc_2026_appearances").fetchone()[0]),
        },
    }
    (out_dir / "external_meta.json").write_text(json.dumps(meta, indent=1))
    return meta, sample


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--include-usage", action="store_true",
                    help="also scan appearances/lineups; slower, but adds WC usage columns")
    args = ap.parse_args()
    meta, sample = build_external_mart(args.start, include_usage=args.include_usage)
    print(f"external mart rows: {meta['rows']}")
    print(sample.to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
