"""Build compact player/market enrichment tables from open Transfermarkt data.

Run:
  .venv/Scripts/python external_data.py

Outputs live under output/external/ and are ignored by git. This keeps the repo
small while making the player/market layer reproducible.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from fifa_rankings import TEAM_ALIASES
from wc_sim import ROOT

OUT = ROOT / "output" / "external"
FIFA_RANKINGS = ROOT / "data" / "fifa_rankings_latest.csv"
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


def _table_path(name, out_dir):
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"{name}.csv.gz"
    if not dst.exists():
        req = urllib.request.Request(TABLES[name], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            dst.write_bytes(r.read())
    return _sql_path(dst)


def build_external_mart(start="2026-06-01", out_dir=OUT, include_usage=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.sql("INSTALL httpfs; LOAD httpfs;")
    except Exception:
        con.sql("LOAD httpfs;")

    con.sql(f"create or replace temp view players as select * from read_csv_auto('{_table_path('players', out_dir)}')")
    con.sql(f"create or replace temp view national_teams as select * from read_csv_auto('{_table_path('national_teams', out_dir)}')")
    alias_rows = ", ".join(
        f"('{k.replace("'", "''")}', '{v.replace("'", "''")}')" for k, v in TEAM_ALIASES.items()
    )
    con.sql(f"create or replace temp view team_aliases(source, team) as values {alias_rows}")
    if FIFA_RANKINGS.exists():
        con.sql(f"""
            create or replace temp view fifa_rankings as
            select coalesce(a.team, fr.team) as team, fr.fifa_ranking, fr.source_date, fr.source
            from read_csv_auto('{_sql_path(FIFA_RANKINGS)}') fr
            left join team_aliases a on a.source = fr.team
        """)
    else:
        con.sql("""
            create or replace temp view fifa_rankings as
            select null::varchar as team, null::integer as fifa_ranking,
                   null::varchar as source_date, null::varchar as source
        """)
    con.sql("""
        create or replace temp view player_rank as
        select nt.national_team_id, coalesce(nta.team, nt.name) as team, nt.confederation,
               coalesce(fr.fifa_ranking, nt.fifa_ranking) as fifa_ranking,
               nt.fifa_ranking as transfermarkt_fifa_ranking,
               nt.squad_size, nt.average_age, nt.total_market_value,
               p.player_id, p.name as player, p.position, p.sub_position, p.foot,
               p.height_in_cm, p.date_of_birth, p.international_caps, p.international_goals,
               p.current_club_name, p.market_value_in_eur,
               row_number() over (
                   partition by nt.national_team_id
                   order by p.market_value_in_eur desc nulls last, p.international_caps desc nulls last
               ) as market_rank
        from national_teams nt
        left join team_aliases nta on nta.source = nt.name
        left join fifa_rankings fr on fr.team = coalesce(nta.team, nt.name)
        left join players p on p.current_national_team_id = nt.national_team_id
        where nt.last_season >= 2025
    """)
    con.sql("""
        create or replace temp view player_pool as
        select team, player_id, player, position, sub_position, foot, height_in_cm,
               date_diff('year', try_cast(date_of_birth as date), current_date) as age,
               international_caps, international_goals, current_club_name,
               market_value_in_eur, market_rank,
               case
                   when position = 'Goalkeeper' then 'GK'
                   when position = 'Defender' then 'DF'
                   when position = 'Midfield' then 'MF'
                   when position = 'Attack' then 'FW'
                   else 'UNK'
               end as pos_group
        from player_rank
        where player_id is not null and market_rank <= 23
    """)
    con.sql("""
        create or replace temp view team_chemistry as
        with counts as (
            select team, count(*) as top23_count,
                   sum(pos_group = 'GK') as gk_count,
                   sum(pos_group = 'DF') as df_count,
                   sum(pos_group = 'MF') as mf_count,
                   sum(pos_group = 'FW') as fw_count,
                   avg(age) as top23_avg_age,
                   stddev_pop(age) as top23_age_std,
                   avg(case when lower(foot) = 'left' then 1.0 else 0.0 end) as left_foot_share,
                   count(distinct current_club_name) as distinct_clubs
            from player_pool
            group by team
        ), clubs as (
            select team, max(club_players) as max_same_club_players
            from (
                select team, current_club_name, count(*) as club_players
                from player_pool
                where current_club_name is not null
                group by team, current_club_name
            )
            group by team
        ), scored as (
            select c.*,
                   coalesce(cl.max_same_club_players, 1) / nullif(c.top23_count, 0) as same_club_share,
                   0.20 * least(c.gk_count / 2.0, 1.0)
                   + 0.30 * least(c.df_count / 6.0, 1.0)
                   + 0.30 * least(c.mf_count / 6.0, 1.0)
                   + 0.20 * least(c.fw_count / 4.0, 1.0) as position_balance,
                   greatest(0.0, 1.0 - abs(coalesce(c.left_foot_share, 0.3) - 0.3) / 0.3) as foot_balance,
                   greatest(0.0, 1.0 - least(coalesce(c.top23_age_std, 8.0), 8.0) / 8.0) as age_cohesion
            from counts c
            left join clubs cl using(team)
        )
        select *,
               0.55 * position_balance + 0.20 * age_cohesion
               + 0.15 * foot_balance + 0.10 * same_club_share as chemistry_score
        from scored
    """)
    con.sql("""
        create or replace temp view team_strength_base as
        select team, confederation, fifa_ranking, squad_size, average_age,
               max(total_market_value) as tm_total_market_value,
               count(player_id) as current_nt_players,
               sum(coalesce(market_value_in_eur, 0)) as listed_market_value,
               sum(case when market_rank <= 11 then coalesce(market_value_in_eur, 0) else 0 end) as top11_market_value,
               sum(case when market_rank <= 23 then coalesce(market_value_in_eur, 0) else 0 end) as top23_market_value,
               sum(coalesce(international_caps, 0)) as squad_caps,
               sum(coalesce(international_goals, 0)) as squad_goals,
               max(tc.chemistry_score) as chemistry_score,
               max(tc.position_balance) as position_balance,
               max(tc.foot_balance) as foot_balance,
               max(tc.same_club_share) as same_club_share,
               max(tc.top23_avg_age) as top23_avg_age,
               max(tc.top23_age_std) as top23_age_std
        from player_rank
        left join team_chemistry tc using(team)
        group by all
    """)
    con.sql("""
        create or replace temp view team_strength as
        select * from team_strength_base
        union all
        select fr.team,
               null::varchar as confederation,
               fr.fifa_ranking,
               null::integer as squad_size,
               null::double as average_age,
               null::double as tm_total_market_value,
               null::bigint as current_nt_players,
               null::double as listed_market_value,
               null::double as top11_market_value,
               null::double as top23_market_value,
               null::double as squad_caps,
               null::double as squad_goals,
               null::double as chemistry_score,
               null::double as position_balance,
               null::double as foot_balance,
               null::double as same_club_share,
               null::double as top23_avg_age,
               null::double as top23_age_std
        from fifa_rankings fr
        left join team_strength_base ts on lower(ts.team) = lower(fr.team)
        where fr.team is not null and ts.team is null
    """)
    if include_usage:
        con.sql(f"create or replace temp view games as select * from read_csv_auto('{_table_path('games', out_dir)}')")
        con.sql(f"create or replace temp view appearances as select * from read_csv_auto('{_table_path('appearances', out_dir)}')")
        con.sql(
            "create or replace temp view game_lineups_raw as "
            f"select * from read_csv_auto('{_table_path('game_lineups', out_dir)}', "
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
    for view in ("player_pool", "team_chemistry", "team_strength", "fiwc_2026_appearances",
                 "fiwc_2026_starts", "project_team_enrichment"):
        con.sql(f"copy (select * from {view}) to '{_sql_path(out_dir / (view + '.csv'))}' (header, delimiter ',')")
    sample = con.sql("""
        select team, fifa_ranking, current_nt_players, top11_market_value, top23_market_value,
               chemistry_score, fiwc_minutes, fiwc_player_goals
        from project_team_enrichment
        order by top23_market_value desc nulls last
        limit 12
    """).fetchdf()
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "dcaribou/transfermarkt-datasets + FIFA live rankings",
        "source_base": BASE,
        "fifa_rankings": str(FIFA_RANKINGS) if FIFA_RANKINGS.exists() else None,
        "start": start,
        "include_usage": include_usage,
        "rows": {
            "team_strength": int(con.sql("select count(*) from team_strength").fetchone()[0]),
            "fifa_only_team_strength": int(con.sql("""
                select count(*) from team_strength ts
                where ts.current_nt_players is null and ts.fifa_ranking is not null
            """).fetchone()[0]),
            "player_pool": int(con.sql("select count(*) from player_pool").fetchone()[0]),
            "team_chemistry": int(con.sql("select count(*) from team_chemistry").fetchone()[0]),
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
