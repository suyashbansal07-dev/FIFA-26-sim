"""Self-checks for wc_sim pure math. Run: .venv/Scripts/python test_wc_sim.py"""
import numpy as np
import pandas as pd
from scipy.stats import poisson

from wc_sim import MAX_GOALS, Simulator, dc_grid, decay_weights, markets, match_rates, shootout_rates


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
    lam_neutral, _ = match_rates(atk, dfn, hfa, "A", "B", "Elsewhere")
    lam_home, mu_home = match_rates(atk, dfn, hfa, "A", "B", "A")
    assert abs(lam_home / lam_neutral - np.exp(hfa)) < 1e-12
    _, mu_neutral = match_rates(atk, dfn, hfa, "A", "B", "Elsewhere")
    assert mu_home == mu_neutral, "away team must not get home advantage"


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


def test_whatif_rejects_impossible_r16_override():
    from server import _validate_overrides
    bracket = {"r16": [{"id": "R16-1", "home": "Canada", "away": "Morocco"}]}
    assert _validate_overrides(bracket, {"R16-1": "Canada"}) == []
    err = _validate_overrides(bracket, {"R16-1": "Brazil"})
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


def test_stronger_team_advances_more():
    atk = {"Strong": 0.8, "Weak": -0.8}
    dfn = {"Strong": -0.5, "Weak": 0.5}
    sim = Simulator(atk, dfn, 0.25, -0.1, np.random.default_rng(1))
    wins = sum(sim.play("Strong", "Weak", "Neutral") == "Strong" for _ in range(2000))
    assert wins / 2000 > 0.75, f"strong team only won {wins/2000:.1%}"


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"ok  {fn.__name__}")
    print("all checks passed")
