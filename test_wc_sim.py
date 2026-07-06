"""Self-checks for wc_sim pure math. Run: .venv/Scripts/python test_wc_sim.py"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from tempfile import TemporaryDirectory
from pathlib import Path

from wc_sim import (MAX_GOALS, Simulator, dc_grid, decay_weights, markets, match_rates,
                    run_tournament, shootout_rates)


def test_tau_matches_spec():
    lam, mu, rho = 1.4, 1.1, -0.12
    base = np.outer(poisson.pmf(np.arange(MAX_GOALS), lam), poisson.pmf(np.arange(MAX_GOALS), mu))
    expect = base.copy()
    expect[0, 0] *= 1 - lam * mu * rho
    expect[0, 1] *= 1 + lam * rho
    expect[1, 0] *= 1 + mu * rho
    expect[1, 1] *= 1 - rho
    expect /= expect.sum()
    assert np.allclose(dc_grid(lam, mu, rho), expect), "tau correction deviates from spec 2.2"


def test_rho_zero_is_independent_poisson():
    g = dc_grid(1.3, 0.9, 0.0)
    base = np.outer(poisson.pmf(np.arange(MAX_GOALS), 1.3), poisson.pmf(np.arange(MAX_GOALS), 0.9))
    assert np.allclose(g, base / base.sum())


def test_grid_and_markets_sum_to_one():
    g = dc_grid(1.7, 1.2, -0.08)
    h, d, a, o25, top = markets(g)
    assert abs(g.sum() - 1) < 1e-12
    assert abs(h + d + a - 1) < 1e-9
    assert 0 < o25 < 1 and len(top) == 3


def test_host_advantage_applies_only_at_home_venue():
    atk, dfn, hfa = {"A": 0.3, "B": 0.1}, {"A": -0.2, "B": -0.1}, 0.25
    lam_neutral, _ = match_rates(atk, dfn, hfa, "A", "B", "Elsewhere", goal_scale=1.0)
    lam_home, mu_home = match_rates(atk, dfn, hfa, "A", "B", "A", goal_scale=1.0)
    assert abs(lam_home / lam_neutral - np.exp(hfa)) < 1e-12
    _, mu_neutral = match_rates(atk, dfn, hfa, "A", "B", "Elsewhere", goal_scale=1.0)
    assert mu_home == mu_neutral, "away team must not get home advantage"


def test_external_prior_moves_rates_symmetrically():
    atk, dfn = {"A": 0.0, "B": 0.0}, {"A": 0.0, "B": 0.0}
    base = match_rates(atk, dfn, 0.0, "A", "B", "", goal_scale=1.0)
    shifted = match_rates(atk, dfn, 0.0, "A", "B", "", goal_scale=1.0,
                          external_strength={"A": 1.0, "B": -1.0}, external_weight=0.05)
    assert base == (1.0, 1.0)
    assert shifted[0] > 1.0 and shifted[1] < 1.0


def test_external_strength_uses_rank_only_fallback_without_fake_player_data():
    from external_signals import build_external_strength
    rows = pd.DataFrame([
        {"team": "A", "top23_market_value": 1_000_000_000, "fifa_ranking": 1,
         "squad_caps": 1000, "squad_goals": 200, "chemistry_score": 0.8},
        {"team": "B", "top23_market_value": 100_000_000, "fifa_ranking": 100,
         "squad_caps": 200, "squad_goals": 20, "chemistry_score": 0.5},
        {"team": "Cape Verde", "top23_market_value": np.nan, "fifa_ranking": 67,
         "squad_caps": np.nan, "squad_goals": np.nan, "chemistry_score": np.nan},
    ])
    strength = build_external_strength(rows)
    assert set(strength) == {"A", "B", "Cape Verde"}
    assert strength["A"] > strength["B"]
    assert np.isfinite(strength["Cape Verde"])
    assert strength["B"] < strength["Cape Verde"] < strength["A"]


def test_external_strength_uses_quality_depth_and_chemistry():
    from external_signals import build_external_strength
    rows = pd.DataFrame([
        {"team": "Balanced", "top11_market_value": 500_000_000,
         "top23_market_value": 900_000_000, "fifa_ranking": 10,
         "squad_caps": 800, "squad_goals": 120, "chemistry_score": 0.85,
         "position_balance": 1.0, "same_club_share": 0.2},
        {"team": "Thin", "top11_market_value": 500_000_000,
         "top23_market_value": 550_000_000, "fifa_ranking": 10,
         "squad_caps": 800, "squad_goals": 120, "chemistry_score": 0.55,
         "position_balance": 0.6, "same_club_share": 0.05},
    ])
    strength = build_external_strength(rows)
    assert strength["Balanced"] > strength["Thin"]


def test_external_strength_uses_star_x_factor_without_team_exceptions():
    from external_signals import build_external_strength
    common = {"top11_market_value": 500_000_000, "top23_market_value": 700_000_000,
              "fifa_ranking": 20, "squad_caps": 500, "squad_goals": 80,
              "chemistry_score": 0.7, "position_balance": 0.8, "same_club_share": 0.1}
    rows = pd.DataFrame([
        {"team": "OneStar", **common, "top1_market_value": 200_000_000,
         "top3_market_value": 300_000_000, "top_attacker_market_value": 200_000_000},
        {"team": "Flat", **common, "top1_market_value": 60_000_000,
         "top3_market_value": 170_000_000, "top_attacker_market_value": 60_000_000},
    ])
    strength = build_external_strength(rows)
    assert strength["OneStar"] > strength["Flat"]


def test_fifa_live_ranking_rows_are_canonicalized():
    from fifa_rankings import rows_from_payload
    payload = {"Results": [
        {"TeamName": [{"Locale": "en-GB", "Description": "France"}], "Rank": 1,
         "IdCountry": "FRA", "TotalPoints": 1925.861, "PrevRank": 3,
         "RankingMovement": 2, "ConfederationName": "UEFA"},
        {"TeamName": [{"Locale": "en-GB", "Description": "Cabo Verde"}], "Rank": 64,
         "IdCountry": "CPV", "TotalPoints": 1402.966, "PrevRank": 67,
         "RankingMovement": 3, "ConfederationName": "CAF"},
        {"TeamName": [{"Locale": "en-GB", "Description": "USA"}], "Rank": 17,
         "IdCountry": "USA", "TotalPoints": 1647.0, "PrevRank": 17,
         "RankingMovement": 0, "ConfederationName": "CONCACAF"},
    ]}
    rows = rows_from_payload(payload, source_date="2026-07-05")
    names = {r["team"]: r for r in rows}
    assert names["France"]["fifa_ranking"] == 1
    assert names["Cape Verde"]["country_code"] == "CPV"
    assert names["United States"]["country_code"] == "USA"


def test_form_strength_rewards_opponent_adjusted_recent_run():
    from form_signals import build_recent_form_strength, form_rate_adjustment
    rows = pd.DataFrame([
        {"date": "2026-06-01", "home_team": "Underdog", "away_team": "Elite",
         "home_score": 1, "away_score": 1},
        {"date": "2026-06-05", "home_team": "Underdog", "away_team": "Strong",
         "home_score": 2, "away_score": 1},
        {"date": "2026-06-01", "home_team": "Favorite", "away_team": "Weak",
         "home_score": 1, "away_score": 0},
        {"date": "2026-06-05", "home_team": "Favorite", "away_team": "Weak",
         "home_score": 0, "away_score": 0},
    ])
    external = {"Elite": 1.8, "Strong": 1.0, "Favorite": 0.7,
                "Underdog": -0.7, "Weak": -1.0}
    strength, meta = build_recent_form_strength(rows, external_strength=external)
    assert meta["rows"] >= 2
    assert strength["Underdog"] > strength["Favorite"]
    assert form_rate_adjustment("Underdog", "Favorite", strength, 0.04) > 0


def test_form_strength_uses_prior_match_stat_pressure():
    from form_signals import build_recent_form_strength
    rows = pd.DataFrame([
        {"date": "2026-06-01", "home_team": "Pressure", "away_team": "Passive",
         "home_score": 0, "away_score": 0},
        {"date": "2026-06-05", "home_team": "Passive", "away_team": "Pressure",
         "home_score": 0, "away_score": 0},
    ])
    features = pd.DataFrame([
        {"date": "2026-06-01", "home_team": "Pressure", "away_team": "Passive",
         "home_shots": 18, "away_shots": 4, "home_sot": 7, "away_sot": 1,
         "home_corners": 9, "away_corners": 2, "home_possession": 64, "away_possession": 36},
        {"date": "2026-06-05", "home_team": "Passive", "away_team": "Pressure",
         "home_shots": 5, "away_shots": 17, "home_sot": 1, "away_sot": 6,
         "home_corners": 1, "away_corners": 8, "home_possession": 39, "away_possession": 61},
    ])
    strength, meta = build_recent_form_strength(rows, features=features, min_matches=2)
    assert meta["rows"] == 2
    assert strength["Pressure"] > strength["Passive"]


def test_form_prior_moves_rates_after_external_prior():
    atk, dfn = {"A": 0.0, "B": 0.0}, {"A": 0.0, "B": 0.0}
    lam, mu = match_rates(atk, dfn, 0.0, "A", "B", "", goal_scale=1.0,
                          form_strength={"A": 1.0, "B": -1.0}, form_weight=0.04)
    assert lam > 1.0 and mu < 1.0


def test_decay_weights_and_friendly_downweight():
    dates = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-01", "2024-01-01"]))
    friendly = pd.Series([False, True, False])
    w = decay_weights(dates, half_life_days=365.25 * 2, friendly_mask=friendly, friendly_weight=0.5)
    assert w[0] == 1.0 and abs(w[1] - 0.5) < 1e-12, "friendly must be downweighted"
    assert abs(w[2] - 0.5) < 0.01, "match one half-life ago must weigh ~0.5"


def test_shootout_rates_shrinkage():
    s = pd.DataFrame({"home_team": ["A", "A", "A", "B"], "away_team": ["B", "C", "B", "C"],
                      "winner": ["A", "A", "A", "B"]})
    r = shootout_rates(s)
    assert 0 < min(r.values()) and max(r.values()) < 1
    assert r["A"] > r["C"], "3-for-3 team must rate above 0-for-2 team"
    assert abs(r["A"] - 8 / 13) < 1e-12, "Beta(5,5) shrinkage: (3+5)/(3+10)"


def test_draws_resolve_via_et_and_pens():
    # near-zero rates force 0-0 in 90' and ET -> pens decide via historical rates
    atk, dfn = {"A": -6.0, "B": -6.0}, {"A": 0.0, "B": 0.0}
    sim = Simulator(atk, dfn, 0.0, 0.0, np.random.default_rng(7), pens={"A": 0.9, "B": 0.1})
    wins = sum(sim.play("A", "B", "Neutral") == "A" for _ in range(800))
    assert 0.8 < wins / 800 <= 1.0, f"pens specialist won only {wins/800:.1%}"


def test_rps_known_values():
    from backtest import rps
    assert rps(np.array([1.0, 0.0, 0.0]), 0) == 0.0, "perfect forecast must score 0"
    # uniform forecast, home win: ((1/3-1)^2 + (2/3-1)^2)/2 = 5/18
    assert abs(rps(np.full(3, 1 / 3), 0) - 5 / 18) < 1e-12
    good, bad = np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.2, 0.7])
    assert rps(good, 0) < rps(bad, 0), "RPS must reward the sharper forecast"


def test_backtest_scoreline_calibration_metrics_are_recorded():
    from backtest import _score_rows
    rows = pd.DataFrame([{"home_team": "A", "away_team": "B", "home_score": 2,
                          "away_score": 1, "neutral": True, "outcome": 0}])
    keys = ("rps", "brier", "logloss", "fav_p", "fav_hit", "uniform", "freq",
            "pred_goals", "actual_goals", "pred_over25", "actual_over25",
            "scoreline_logloss", "score_top1", "score_top3", "top_low_score")
    sink = {k: [] for k in keys} | {"skipped": 0, "_freq": np.full(3, 1 / 3)}
    _score_rows(rows, {"A": 0.1, "B": -0.1}, {"A": -0.1, "B": 0.1},
                0.0, -0.05, sink, goal_scale=1.0)
    assert sink["skipped"] == 0
    assert sink["actual_goals"] == [3]
    assert sink["pred_goals"][0] > 0
    assert 0 <= sink["pred_over25"][0] <= 1
    assert sink["scoreline_logloss"][0] > 0


def test_whatif_rejects_impossible_r16_override():
    from server import _validate_overrides
    bracket = {"r16": [{"id": "R16-1", "home": "Canada", "away": "Morocco"}],
               "qf": [], "sf": [], "final": {"id": "F", "from": ["SF-1", "SF-2"]}}
    assert _validate_overrides(bracket, {"R16-1": "Canada"}, {}) == []
    err = _validate_overrides(bracket, {"R16-1": "Brazil"}, {})
    assert err and "R16-1" in err[0] and "Canada" in err[0]


def test_confederation_inference_uses_competition_evidence():
    from diagnostics import infer_confederations
    rows = pd.DataFrame({
        "home_team": ["France", "United States"],
        "away_team": ["Spain", "Mexico"],
        "home_score": [1, 2],
        "away_score": [0, 1],
        "tournament": ["UEFA Nations League", "CONCACAF Gold Cup"],
    })
    meta = infer_confederations(rows)
    assert meta["France"]["confed"] == "UEFA"
    assert meta["Mexico"]["confed"] == "CONCACAF"


def test_forward_loop_settles_only_pre_match_forecasts():
    from forward_loop import record_payload_forecasts, settle_forward_forecasts
    payload = {"meta": {"generated": "2026-07-01T00:00:00+00:00", "half_life_days": 550,
                        "friendly_weight": 1, "hfa": 0.2, "rho": -0.08},
               "fixtures": [{"id": "R16-1", "date": "2026-07-04", "home": "Canada",
                             "away": "Morocco", "venue": "United States", "played": False,
                             "p_home": 0.2, "p_draw": 0.3, "p_away": 0.5, "over25": 0.4}]}
    matches = pd.DataFrame({"date": pd.to_datetime(["2026-07-04"]),
                            "home_team": ["Canada"], "away_team": ["Morocco"],
                            "home_score": [1], "away_score": [0]})
    with TemporaryDirectory() as d:
        ledger = Path(d) / "ledger.jsonl"
        report = Path(d) / "calibration.json"
        record_payload_forecasts(payload, ledger, now=pd.Timestamp("2026-07-01", tz="UTC").to_pydatetime())
        settled = settle_forward_forecasts(matches, ledger, report)
        assert settled["settled"] == 1 and settled["pending"] == 0 and settled["late_excluded"] == 0
        record_payload_forecasts(payload, ledger, now=pd.Timestamp("2026-07-05", tz="UTC").to_pydatetime())
        settled = settle_forward_forecasts(matches, ledger, report)
        assert settled["settled"] == 1 and settled["late_excluded"] == 1
        assert settled["calibration_policy"]["action"] == "hold"


def test_espn_topup_fills_pending_fixture_instead_of_duplicating():
    from fetch_data import espn_topup
    matches = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-05"), "home_team": "Mexico", "away_team": "England",
         "home_score": np.nan, "away_score": np.nan, "tournament": "FIFA World Cup",
         "city": "Mexico City", "country": "Mexico", "neutral": False},
        {"date": pd.Timestamp("2026-07-04"), "home_team": "Canada", "away_team": "Morocco",
         "home_score": 0.0, "away_score": 3.0, "tournament": "FIFA World Cup",
         "city": "Houston", "country": "United States", "neutral": True},
    ])
    event = {"date": "2026-07-06T00:30Z", "status": {"type": {"name": "STATUS_FULL_TIME"}},
             "competitions": [{"venue": {"address": {"city": "Mexico City", "country": "Mexico"}},
                               "competitors": [
                                   {"homeAway": "home", "score": "2",
                                    "team": {"displayName": "Mexico"}},
                                   {"homeAway": "away", "score": "3",
                                    "team": {"displayName": "England"}},
                               ]}]}
    out, pens, n = espn_topup(matches, pd.DataFrame(), events=[event], today="2026-07-06")
    assert n == 1 and len(out) == 2
    row = out[out["home_team"].eq("Mexico") & out["away_team"].eq("England")].iloc[0]
    assert row["home_score"] == 2 and row["away_score"] == 3


def test_espn_topup_polls_same_day_after_first_result():
    import io
    import json
    import fetch_data
    matches = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-06"), "home_team": "A", "away_team": "B",
         "home_score": 1.0, "away_score": 0.0, "tournament": "FIFA World Cup",
         "city": "X", "country": "Y", "neutral": True},
        {"date": pd.Timestamp("2026-07-06"), "home_team": "C", "away_team": "D",
         "home_score": np.nan, "away_score": np.nan, "tournament": "FIFA World Cup",
         "city": "X", "country": "Y", "neutral": True},
    ])
    event = {"date": "2026-07-06T22:00Z", "status": {"type": {"name": "STATUS_FULL_TIME"}},
             "competitions": [{"venue": {"address": {"city": "X", "country": "Y"}},
                               "competitors": [
                                   {"homeAway": "home", "score": "2", "team": {"displayName": "C"}},
                                   {"homeAway": "away", "score": "1", "team": {"displayName": "D"}},
                               ]}]}

    class Response(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    old = fetch_data.urllib.request.urlopen
    calls = []
    try:
        def fake_open(url, timeout=30):
            calls.append(url)
            return Response(json.dumps({"events": [event]}).encode())
        fetch_data.urllib.request.urlopen = fake_open
        out, _, n = fetch_data.espn_topup(matches, pd.DataFrame(), today="2026-07-06")
        row = out[out["home_team"].eq("C")].iloc[0]
        assert calls and n == 1
        assert row["home_score"] == 2 and row["away_score"] == 1
    finally:
        fetch_data.urllib.request.urlopen = old


def test_server_applies_forward_calibration_once():
    import json
    import server
    with TemporaryDirectory() as d:
        root = Path(d)
        old_report, old_applied = server.FORWARD_CALIBRATION, server.FORWARD_CALIBRATION_APPLIED
        old_weight = server.CFG["external_weight"]
        server.FORWARD_CALIBRATION = root / "forward_calibration.json"
        server.FORWARD_CALIBRATION_APPLIED = root / "forward_calibration_applied.json"
        try:
            server.CFG["external_weight"] = 0.12
            server.FORWARD_CALIBRATION.write_text(json.dumps({
                "generated": "g-hold",
                "calibration_policy": {"action": "hold", "reason": "need 12", "settled": 2},
            }))
            out = server._apply_forward_calibration()
            assert not out["applied"] and server.CFG["external_weight"] == 0.12

            server.FORWARD_CALIBRATION.write_text(json.dumps({
                "generated": "g-apply",
                "calibration_policy": {"action": "allow_slightly_more_prior_confidence",
                                       "settled": 12},
            }))
            out = server._apply_forward_calibration()
            assert out["applied"] and out["before"] == 0.12 and out["after"] == 0.13
            again = server._apply_forward_calibration()
            assert not again["applied"] and again["reason"] == "already applied"
            assert server.CFG["external_weight"] == 0.13
        finally:
            server.FORWARD_CALIBRATION = old_report
            server.FORWARD_CALIBRATION_APPLIED = old_applied
            server.CFG["external_weight"] = old_weight


def test_match_feature_extracts_stats_and_xg_from_espn_shapes():
    from match_features import attach_features, feature_row_from_event, feature_coverage
    event = {
        "id": "1", "date": "2026-07-04T00:00Z",
        "status": {"type": {"name": "STATUS_FULL_TIME"}},
        "competitions": [{"venue": {"address": {"country": "United States"}},
                          "competitors": [
                              {"homeAway": "home", "team": {"displayName": "United States"},
                               "statistics": [{"name": "totalShots", "displayValue": "10"},
                                              {"name": "shotsOnTarget", "displayValue": "4"}]},
                              {"homeAway": "away", "team": {"displayName": "Mexico"},
                               "statistics": [{"name": "totalShots", "displayValue": "8"},
                                              {"name": "shotsOnTarget", "displayValue": "2"}]},
                          ]}],
    }
    summary = {"leaders": [
        {"team": {"displayName": "United States"}, "leaders": [
            {"name": "saves", "leaders": [{"statistics": [
                {"name": "expectedGoalsConceded", "value": 0.7}]}]}]},
        {"team": {"displayName": "Mexico"}, "leaders": [
            {"name": "saves", "leaders": [{"statistics": [
                {"name": "expectedGoalsConceded", "value": 1.4}]}]}]},
    ]}
    row = feature_row_from_event(event, summary)
    assert row["home_team"] == "United States" and row["away_team"] == "Mexico"
    assert row["home_shots"] == 10 and row["away_sot"] == 2
    assert row["home_xg"] == 1.4 and row["away_xg"] == 0.7
    records = [{"date": "2026-07-04", "home_team": "United States", "away_team": "Mexico"}]
    attach_features(records, pd.DataFrame([row]).assign(date=pd.Timestamp("2026-07-04")))
    assert records[0]["has_match_features"] and records[0]["home_xg"] == 1.4
    assert feature_coverage(records)["coverage"]["xg"] == 1.0


def test_external_payload_enriches_ratings():
    import json
    import server
    payload = {"meta": {}, "ratings": [{"team": "Canada", "attack": 0.1, "defence": -0.1}]}
    with TemporaryDirectory() as d:
        old = server.EXTERNAL_DIR
        server.EXTERNAL_DIR = Path(d)
        try:
            pd.DataFrame([{"team": "Canada", "confederation": "CONCACAF", "fifa_ranking": 30,
                           "current_nt_players": 37, "top11_market_value": 165500000,
                           "top23_market_value": 199500000, "top_player": "Alphonso Davies",
                           "top_player_position": "Defender", "top1_market_value": 50000000,
                           "top3_market_value": 95000000, "top_attacker_market_value": 12000000,
                           "squad_caps": 1184,
                           "squad_goals": 158, "fiwc_player_appearances": 44,
                           "fiwc_minutes": 2700, "fiwc_player_goals": 7,
                           "fiwc_assists": 4, "fiwc_yellow_cards": 2,
                           "fiwc_red_cards": 0}]).to_csv(Path(d) / "project_team_enrichment.csv", index=False)
            (Path(d) / "external_meta.json").write_text(json.dumps({"source": "test", "generated": "now"}))
            out = server._attach_external(payload)
            assert out["external"]["present"] and out["meta"]["external_data"]["rows"] == 1
            assert out["ratings"][0]["fifa_ranking"] == 30
            assert out["ratings"][0]["top23_market_value"] == 199500000
            assert out["ratings"][0]["top_player"] == "Alphonso Davies"
            assert out["external"]["teams"][0]["top1_market_value"] == 50000000
            assert out["external"]["teams"][0]["fiwc_minutes"] == 2700
            assert out["external"]["teams"][0]["fiwc_assists"] == 4
        finally:
            server.EXTERNAL_DIR = old


def test_load_state_attaches_external_strength_after_reload():
    import json
    import server
    with TemporaryDirectory() as d:
        root = Path(d)
        old_state_file, old_external_dir = server.STATE_FILE, server.EXTERNAL_DIR
        old_form_loader = server._load_form_strength
        old_state = {k: v for k, v in server.STATE.items()}
        server.STATE_FILE = root / "state.json"
        server.EXTERNAL_DIR = root / "external"
        server.EXTERNAL_DIR.mkdir()
        try:
            pd.DataFrame([{"team": "Canada", "fifa_ranking": 16,
                           "top23_market_value": 199500000, "squad_caps": 1184,
                           "squad_goals": 158, "chemistry_score": 0.7}]).to_csv(
                server.EXTERNAL_DIR / "project_team_enrichment.csv", index=False)
            (server.EXTERNAL_DIR / "external_meta.json").write_text(json.dumps({"source": "test"}))
            server.STATE_FILE.write_text(json.dumps({
                "payload": {"meta": {}, "bracket": [{"team": "Canada", "bronze": 0.0}],
                            "ratings": [{"team": "Canada"}],
                            "verdict": {"champion": "Canada", "matches": []}},
                "pens": {},
                "params": {"attack": {"Canada": 0.1}, "defence": {"Canada": -0.1},
                           "hfa": 0.2, "rho": -0.08},
            }))
            server.STATE.update({"payload": None, "external_strength": {}, "form_strength": {}})
            server._load_form_strength = lambda: ({}, {})
            assert server.load_state()
            meta = server.STATE["payload"]["meta"]["external_data"]
            assert meta["strength_rows"] == 1
            assert server.STATE["payload"]["ratings"][0]["fifa_ranking"] == 16
        finally:
            server.STATE_FILE = old_state_file
            server.EXTERNAL_DIR = old_external_dir
            server._load_form_strength = old_form_loader
            server.STATE.clear()
            server.STATE.update(old_state)


def test_load_state_rejects_stale_no_bronze_payload():
    import json
    import server
    with TemporaryDirectory() as d:
        old_state_file = server.STATE_FILE
        old_state = {k: v for k, v in server.STATE.items()}
        server.STATE_FILE = Path(d) / "state.json"
        try:
            server.STATE_FILE.write_text(json.dumps({
                "payload": {"meta": {}, "bracket": [{"team": "Canada"}], "ratings": []},
                "pens": {},
                "params": {"attack": {}, "defence": {}, "hfa": 0.0, "rho": 0.0},
            }))
            assert not server.load_state()
            assert server.STATE.get("payload") is old_state.get("payload")
        finally:
            server.STATE_FILE = old_state_file
            server.STATE.clear()
            server.STATE.update(old_state)


def test_case_pre_match_form_prior_excludes_current_match():
    import server
    old_weight = server.CFG["form_weight"]
    old_state = {k: v for k, v in server.STATE.items()}
    rows = pd.DataFrame([
        {"date": "2026-07-01", "home_team": "A", "away_team": "B",
         "home_score": 0, "away_score": 0},
        {"date": "2026-07-02", "home_team": "B", "away_team": "A",
         "home_score": 0, "away_score": 0},
        {"date": "2026-07-05", "home_team": "A", "away_team": "B",
         "home_score": 5, "away_score": 0},
    ])
    try:
        server.CFG["form_weight"] = 0.04
        server.STATE["external_strength"] = {}
        pre = server._pre_match_form_prior("A", "B", "2026-07-05", df=rows, features=pd.DataFrame())
        post = server._pre_match_form_prior("A", "B", "2026-07-06", df=rows, features=pd.DataFrame())
        assert pre["home_strength"] == 0.0 and pre["away_strength"] == 0.0
        assert post["home_strength"] > post["away_strength"]
    finally:
        server.CFG["form_weight"] = old_weight
        server.STATE.clear()
        server.STATE.update(old_state)


def test_state_refresh_and_freshness_detect_new_day_staleness():
    import server
    assert server._state_needs_refresh(
        {"generated": "2026-07-05T08:00:00+00:00"}, today="2026-07-06")
    assert not server._state_needs_refresh(
        {"generated": "2026-07-06T01:00:00+00:00"}, today="2026-07-06")
    bracket = {
        "r16": [{"id": "R16-1", "home": "A", "away": "B",
                 "venue_country": "X", "date": "2026-07-05"}],
        "qf": [], "sf": [], "final": {"id": "F", "from": ["SF-1", "SF-2"]},
    }
    f = server._freshness_meta({"newest_result": "2026-07-04"}, bracket, {}, today="2026-07-06")
    assert f["stale"] and f["overdue_unplayed_slots"] == ["R16-1"]
    assert f["result_lag_days"] == 2


def test_forward_safe_context_uses_only_prior_matches():
    from feature_context import add_forward_safe_context
    rows = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-05"]),
        "home_team": ["A", "A"],
        "away_team": ["B", "C"],
        "home_score": [2, 1],
        "away_score": [0, 1],
    })
    out = add_forward_safe_context(rows)
    first = out.iloc[0]
    second = out.iloc[1]
    assert first.home_matches_seen == 0 and first.away_matches_seen == 0
    assert second.home_matches_seen == 1
    assert second.home_days_rest == 4
    assert second.home_ppg_recent == 3


def test_stronger_team_advances_more():
    atk = {"Strong": 0.8, "Weak": -0.8}
    dfn = {"Strong": -0.5, "Weak": 0.5}
    sim = Simulator(atk, dfn, 0.25, -0.1, np.random.default_rng(1))
    wins = sum(sim.play("Strong", "Weak", "Neutral") == "Strong" for _ in range(2000))
    assert wins / 2000 > 0.75, f"strong team only won {wins/2000:.1%}"


def test_vectorized_tournament_sampler_outputs_all_teams():
    atk = {"A": 0.6, "B": -0.2, "C": 0.2, "D": -0.4}
    dfn = {"A": -0.4, "B": 0.1, "C": -0.1, "D": 0.3}
    sim = Simulator(atk, dfn, 0.0, -0.05, np.random.default_rng(2))
    bracket = {
        "r16": [{"id": "R16-1", "home": "A", "away": "B", "venue_country": "Neutral"},
                {"id": "R16-2", "home": "C", "away": "D", "venue_country": "Neutral"}],
        "qf": [],
        "sf": [{"id": "SF-1", "from": ["R16-1", "R16-2"], "venue_country": "Neutral"}],
        "final": {"id": "F", "from": ["SF-1", "SF-1"], "venue_country": "Neutral"},
    }
    probs = run_tournament(sim, bracket, {}, 2000, sampler="antithetic")
    assert set(probs) == {"A", "B", "C", "D"}
    assert abs(sum(p[3] for p in probs.values()) - 1) < 1e-12
    assert probs["A"][0] > probs["B"][0]
    for sampler in ("lhs", "sobol"):
        sim = Simulator(atk, dfn, 0.0, -0.05, np.random.default_rng(2))
        probs = run_tournament(sim, bracket, {}, 256, sampler=sampler)
        assert abs(sum(p[3] for p in probs.values()) - 1) < 1e-12


def test_consensus_modal_and_coherent():
    from consensus import build_consensus
    bracket = {"r16": [{"id": "R16-1"}, {"id": "R16-2"}], "qf": [{"id": "QF-1"}],
               "sf": [{"id": "SF-1"}], "final": {"id": "F"}}
    paths = {"teams": ["A", "B", "C", "D"], "winners": {
        "R16-1": np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0], dtype=np.int16),
        "R16-2": np.array([2, 2, 2, 3, 3, 2, 2, 2, 3, 2], dtype=np.int16),
        "QF-1":  np.array([0, 0, 0, 0, 0, 2, 2, 0, 0, 0], dtype=np.int16),
        "SF-1":  np.array([0, 0, 0, 0, 0, 2, 2, 2, 0, 0], dtype=np.int16),
        "F":     np.array([0, 0, 0, 0, 0, 2, 2, 2, 3, 3], dtype=np.int16),
    }}
    c = build_consensus(paths, bracket, known={"R16-1": "A"})
    assert c["modal_champion"] == {"team": "A", "p": 0.5}
    picks = {p["slot"]: p for p in c["consensus_path"]["picks"]}
    assert picks["F"]["winner"] == "A" and picks["SF-1"]["winner"] == "A"
    assert picks["R16-2"]["winner"] == "C" and picks["R16-2"]["conditional_p"] == 0.6
    assert picks["QF-1"]["slot_p"] == 0.8
    assert picks["R16-1"]["known"] and not picks["F"]["known"]
    assert c["consensus_path"]["joint_support"] == 0.3
    top = c["top_paths"][0]
    assert top["count"] == 3 and top["path"] == {
        "R16-1": "A", "R16-2": "C", "QF-1": "A", "SF-1": "A", "F": "A"}


def test_resolved_fixtures_advance_round_by_round():
    import json
    from pathlib import Path
    from wc_sim import resolved_fixtures
    bracket = json.loads((Path(__file__).parent / "bracket_2026.json").read_text())
    base = resolved_fixtures(bracket, {})
    assert len(base) == 8 and all(f["round"] == "r16" for f in base), "no winners -> R16 only"
    known = {"R16-1": "Morocco", "R16-2": "France"}
    fx = resolved_fixtures(bracket, known)
    qf1 = next((f for f in fx if f["id"] == "QF-1"), None)
    assert qf1 and qf1["home"] == "Morocco" and qf1["away"] == "France" and qf1["round"] == "qf"
    assert len(fx) == 9, "only QF-1 is determined"
    # full R16 + QF winners -> both SFs form
    known = {f"R16-{i}": bracket["r16"][i - 1]["home"] for i in range(1, 9)}
    known.update({f"QF-{i}": "X" for i in range(1, 5)})
    sfs = [f for f in resolved_fixtures(bracket, known) if f["round"] == "sf"]
    assert len(sfs) == 2 and all(f["home"] == "X" and f["away"] == "X" for f in sfs)


def test_whatif_validation_covers_advanced_slots():
    import json
    from pathlib import Path
    from server import _known_with_overrides, _validate_overrides
    bracket = json.loads((Path(__file__).parent / "bracket_2026.json").read_text())
    known = {"R16-1": "Morocco", "R16-2": "France"}
    assert _validate_overrides(bracket, {"QF-1": "France"}, known) == [], \
        "determined QF slot must be pinnable"
    assert _validate_overrides(bracket, {"QF-2": "Spain"}, known), "unformed QF not pinnable"
    assert _validate_overrides(bracket, {"R16-1": "Canada"}, known), "played slot not re-pinnable"
    played_known = {**known, "QF-1": "France", "SF-1": "France"}
    assert _validate_overrides(bracket, {"R16-1": "Canada"}, played_known, counterfactual=True) == []
    cf = _known_with_overrides(bracket, played_known, {"R16-1": "Canada"}, counterfactual=True)
    assert cf["R16-1"] == "Canada"
    assert "QF-1" not in cf and "SF-1" not in cf
    assert _validate_overrides(bracket, {"QF-1": "Brazil"}, known), "non-participant rejected"


def test_run_ensemble_mixes_bootstrap_samples():
    import json
    from pathlib import Path
    from wc_sim import run_ensemble, run_tournament
    bracket = json.loads((Path(__file__).parent / "bracket_2026.json").read_text())
    teams = sorted({t for fx in bracket["r16"] for t in (fx["home"], fx["away"])})
    up = {"attack": {t: 0.1 * i for i, t in enumerate(teams)},
          "defence": {t: 0.0 for t in teams}, "hfa": 0.2, "rho": -0.08}
    down = {"attack": {t: 0.1 * (len(teams) - i) for i, t in enumerate(teams)},
            "defence": {t: 0.0 for t in teams}, "hfa": 0.2, "rho": -0.08}
    probs, paths = run_ensemble([up, down], {}, bracket, {}, 128, "antithetic", seed=5)
    assert len(paths["winners"]["F"]) == 128, "ensemble must simulate all requested paths"
    assert abs(sum(p[3] for p in probs.values()) - 1.0) < 1e-9
    solo = run_tournament(
        Simulator(up["attack"], up["defence"], 0.2, -0.08, np.random.default_rng(5)),
        bracket, {}, 128, "antithetic")
    best_up = max(up["attack"], key=up["attack"].get)
    assert probs[best_up][3] < solo[best_up][3], \
        "mixing an opposing sample must soften the favorite's championship probability"


def test_run_tournament_paths_respect_bracket_tree():
    import json
    from pathlib import Path
    from wc_sim import run_tournament
    bracket = json.loads((Path(__file__).parent / "bracket_2026.json").read_text())
    teams = sorted({t for fx in bracket["r16"] for t in (fx["home"], fx["away"])})
    atk = {t: 0.1 * i for i, t in enumerate(teams)}
    dfn = {t: -0.05 * i for i, t in enumerate(teams)}
    sim = Simulator(atk, dfn, 0.2, -0.08, np.random.default_rng(3))
    probs, paths = run_tournament(sim, bracket, {}, 64, "antithetic", return_paths=True)
    w, names = paths["winners"], paths["teams"]
    for fx in bracket["qf"] + bracket["sf"] + [bracket["final"]]:
        a, b = fx["from"]
        ok = (w[fx["id"]] == w[a]) | (w[fx["id"]] == w[b])
        assert ok.all(), f"{fx['id']} winner must come from {a}/{b}"
    champ_share = sum(p[3] for p in probs.values())
    assert abs(champ_share - 1.0) < 1e-9, "champion probabilities must sum to 1"


def test_verdict_bracket_uses_match_advance_probabilities():
    import json
    from pathlib import Path
    from wc_sim import verdict_bracket
    bracket = json.loads((Path(__file__).parent / "bracket_2026.json").read_text())
    teams = sorted({t for fx in bracket["r16"] for t in (fx["home"], fx["away"])})
    atk = {t: 0.08 * i for i, t in enumerate(teams)}
    dfn = {t: -0.02 * i for i, t in enumerate(teams)}
    sim = Simulator(atk, dfn, 0.0, -0.05, np.random.default_rng(4))
    verdict = verdict_bracket(sim, bracket, {"R16-1": "Morocco"})
    rows = {r["id"]: r for r in verdict["matches"]}
    assert len(verdict["matches"]) == 16
    assert rows["R16-1"]["played"] and rows["R16-1"]["support"] == 1.0
    assert rows["QF-1"]["home"] == "Morocco", "known R16 facts must feed later fixtures"
    assert verdict["champion"] == rows["F"]["winner"]
    assert all(0.5 <= r["support"] < 1.0 for r in verdict["matches"] if not r["played"])


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"ok  {fn.__name__}")
    print("all checks passed")
