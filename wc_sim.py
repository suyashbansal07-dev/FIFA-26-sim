"""2026 FIFA World Cup — time-weighted Dixon-Coles simulator.

Implements "Dixon-Coles Architecture Spec.md":
  - MLE fit via penaltyblog DixonColesGoalModel (attack/defence/home-adv/rho)
    with exponential time-decay weights (spec 2.3)
  - tau-corrected 10x10 scoreline grids and derived markets (spec 2.2, 3.1-3.3)
  - Monte Carlo sim of the remaining knockout bracket in bracket_2026.json,
    conditioned on real results already in the data (spec 3.4)

Usage: python wc_sim.py [--sims 1000000] [--half-life 550] [--years 4] [--seed 26]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson, qmc

from external_signals import (DEFAULT_EXTERNAL_WEIGHT, apply_external_prior,
                              load_external_strength)
from form_signals import DEFAULT_FORM_WEIGHT, apply_form_prior, build_recent_form_strength
from live_signals import DEFAULT_LIVE_WEIGHT, apply_live_prior, build_live_context_strength
from match_features import load_match_features

ROOT = Path(__file__).parent
MAX_GOALS = 10  # grid covers scorelines 0-0 .. 9-9 (spec 3.1)
DEFAULT_SIMS = 1_000_000
DEFAULT_GOAL_SCALE = 1.10
SAMPLERS = ("random", "antithetic", "lhs", "sobol")


# ---------------- data + fit ----------------

def load_matches(years: float) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "matches.csv", parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    # upstream dataset occasionally carries the same match twice with differing
    # city spellings (e.g. Gibraltar-Cayman 2026-06-06) — one vote per match
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    df[["home_score", "away_score"]] = df[["home_score", "away_score"]].astype(int)
    cutoff = df["date"].max() - pd.Timedelta(days=round(365.25 * years))
    df = df[df["date"] >= cutoff]
    # the dataset also carries non-FIFA sides (CONIFA, regional selections) via friendlies;
    # keep only teams that appear in FIFA competitions (WC / WC qualification) in the window
    fifa = df["tournament"].str.contains("FIFA", na=False)
    pool = set(df.loc[fifa, "home_team"]) | set(df.loc[fifa, "away_team"])
    df = df[df["home_team"].isin(pool) & df["away_team"].isin(pool)]
    return df.reset_index(drop=True)


def decay_weights(dates: pd.Series, half_life_days: float,
                  friendly_mask: pd.Series = None, friendly_weight: float = 1.0):
    """W(t)=e^(-xi*t) (spec 2.3), with friendlies optionally downweighted —
    they are a different data-generating process (fewer goals, more draws)."""
    age_days = (dates.max() - dates).dt.days.to_numpy()
    w = np.exp(-math.log(2) / half_life_days * age_days)
    if friendly_mask is not None:
        w = w * np.where(friendly_mask.to_numpy(), friendly_weight, 1.0)
    return w


def fit_model(df: pd.DataFrame, half_life_days: float, friendly_weight: float = 1.0):
    import penaltyblog as pb

    weights = decay_weights(df["date"], half_life_days,
                            df["tournament"].eq("Friendly"), friendly_weight)
    model = pb.models.DixonColesGoalModel(
        df["home_score"], df["away_score"], df["home_team"], df["away_team"],
        weights=weights, neutral_venue=df["neutral"].astype(int),
    )
    model.fit()
    return model


def team_params(model):
    """Extract {team: attack}, {team: defence}, home_advantage, rho from a fitted model."""
    p = model.get_params() if hasattr(model, "get_params") else dict(model.params)
    atk = {k.split("_", 1)[1]: v for k, v in p.items() if k.startswith("attack_")}
    dfn = {k.split("_", 1)[1]: v for k, v in p.items() if k.startswith(("defence_", "defense_"))}
    hfa = next(p[k] for k in ("home_advantage", "hfa") if k in p)
    return atk, dfn, hfa, p["rho"]


# ---------------- inference (spec 2.1, 2.2, 3.2, 3.3) ----------------

def match_rates(atk, dfn, hfa, team_a, team_b, venue_country, goal_scale=DEFAULT_GOAL_SCALE,
                external_strength=None, external_weight=0.0,
                form_strength=None, form_weight=0.0,
                live_strength=None, live_weight=0.0):
    """lambda/mu per spec 2.1; home advantage only when a team plays in its own country
    (hosts USA/Mexico/Canada at this WC — team names equal country names in the data)."""
    lam = goal_scale * math.exp(atk[team_a] + dfn[team_b] + (hfa if team_a == venue_country else 0.0))
    mu = goal_scale * math.exp(atk[team_b] + dfn[team_a] + (hfa if team_b == venue_country else 0.0))
    lam, mu = apply_external_prior(lam, mu, team_a, team_b, external_strength, external_weight)
    lam, mu = apply_form_prior(lam, mu, team_a, team_b, form_strength, form_weight)
    return apply_live_prior(lam, mu, team_a, team_b, live_strength, live_weight)


def dc_grid(lam, mu, rho, n=MAX_GOALS):
    """Bivariate Poisson grid with Dixon-Coles tau correction (spec 2.2)."""
    g = np.outer(poisson.pmf(np.arange(n), lam), poisson.pmf(np.arange(n), mu))
    g[0, 0] *= 1 - lam * mu * rho
    g[0, 1] *= 1 + lam * rho
    g[1, 0] *= 1 + mu * rho
    g[1, 1] *= 1 - rho
    return g / g.sum()  # renormalise away tail truncation


def markets(g):
    """(p_home, p_draw, p_away, p_over25, top3 scorelines) from a grid (spec 3.3)."""
    n = g.shape[0]
    home, away, draw = np.tril(g, -1).sum(), np.triu(g, 1).sum(), np.trace(g)
    over25 = g[np.add.outer(np.arange(n), np.arange(n)) > 2].sum()
    top = sorted(((g[x, y], f"{x}-{y}") for x in range(n) for y in range(n)), reverse=True)[:3]
    return home, draw, away, over25, top


# ---------------- tournament sim (spec 3.4) ----------------

def shootout_rates(shootouts: pd.DataFrame) -> dict:
    """Beta(5,5)-shrunk historical shootout win rate per team (spec 3.4:
    'historically weighted penalty shootout win rate'). ~0.5 for thin histories."""
    apps = pd.concat([shootouts["home_team"], shootouts["away_team"]]).value_counts()
    wins = shootouts["winner"].value_counts()
    return {t: (wins.get(t, 0) + 5.0) / (apps[t] + 10.0) for t in apps.index}


class Simulator:
    def __init__(self, atk, dfn, hfa, rho, rng, pens=None, goal_scale=DEFAULT_GOAL_SCALE,
                 external_strength=None, external_weight=0.0,
                 form_strength=None, form_weight=0.0,
                 live_strength=None, live_weight=0.0):
        self.atk, self.dfn, self.hfa, self.rho, self.rng = atk, dfn, hfa, rho, rng
        self.pens = pens or {}
        self.goal_scale = goal_scale
        self.external_strength = external_strength or {}
        self.external_weight = external_weight
        self.form_strength = form_strength or {}
        self.form_weight = form_weight
        self.live_strength = live_strength or {}
        self.live_weight = live_weight
        self._cache = {}
        self._advance_cache = {}

    def grid_for(self, a, b, venue):
        key = (a, b, venue)
        if key not in self._cache:
            lam, mu = match_rates(self.atk, self.dfn, self.hfa, a, b, venue, self.goal_scale,
                                  self.external_strength, self.external_weight,
                                  self.form_strength, self.form_weight,
                                  self.live_strength, self.live_weight)
            g = dc_grid(lam, mu, self.rho)
            self._cache[key] = (lam, mu, g.ravel(), g)
        return self._cache[key]

    def pens_prob(self, a, b):
        ra, rb = self.pens.get(a, 0.5), self.pens.get(b, 0.5)
        return ra / (ra + rb)

    def advance_prob(self, a, b, venue):
        key = (a, b, venue)
        if key not in self._advance_cache:
            lam, mu, _, g = self.grid_for(a, b, venue)
            h, d, _, _, _ = markets(g)
            et = dc_grid(lam / 3, mu / 3, 0.0)
            eh, ed, _, _, _ = markets(et)
            self._advance_cache[key] = h + d * (eh + ed * self.pens_prob(a, b))
        return self._advance_cache[key]

    def play(self, a, b, venue):
        lam, mu, flat, _ = self.grid_for(a, b, venue)
        x, y = divmod(self.rng.choice(flat.size, p=flat), MAX_GOALS)
        if x != y:
            return a if x > y else b
        # extra time: 30 min of independent Poisson at proportional rates
        ex, ey = self.rng.poisson(lam / 3), self.rng.poisson(mu / 3)
        if ex != ey:
            return a if ex > ey else b
        return a if self.rng.random() < self.pens_prob(a, b) else b


def _uniforms(rng, n_sims, n_draws, sampler="antithetic"):
    if sampler == "random":
        return rng.random((n_sims, n_draws))
    if sampler == "antithetic":
        half = (n_sims + 1) // 2
        u = rng.random((half, n_draws))
        return np.vstack([u, 1.0 - u])[:n_sims]
    seed = int(rng.integers(0, 2**32 - 1))
    if sampler == "lhs":
        return qmc.LatinHypercube(d=n_draws, seed=seed).random(n_sims)
    if sampler == "sobol":
        m = math.ceil(math.log2(max(1, n_sims)))
        return qmc.Sobol(d=n_draws, scramble=True, seed=seed).random_base2(m)[:n_sims]
    raise ValueError(f"unknown sampler: {sampler}")


def _pair_rows(df, home, away):
    return df[(((df["home_team"] == home) & (df["away_team"] == away))
               | ((df["home_team"] == away) & (df["away_team"] == home)))]


def _result_winner(played, shootouts, home, away):
    """Winner of a played match between two teams (either orientation), else None."""
    m = _pair_rows(played, home, away)
    if m.empty:
        return None
    row = m.iloc[-1]
    if row["home_score"] != row["away_score"]:
        return row["home_team"] if row["home_score"] > row["away_score"] else row["away_team"]
    s = _pair_rows(shootouts, home, away)
    return s.iloc[-1]["winner"] if not s.empty else None


def known_winners(bracket, played, shootouts):
    """Real-world winners for knockout slots already decided (manual overrides win).
    Iterates with resolved_fixtures so later rounds resolve as soon as their feeder
    winners identify the participants — QF/SF/final results auto-consume too."""
    known = dict(bracket.get("manual_results", {}))
    ko = played[played["date"] >= "2026-06-28"]
    pens = shootouts[shootouts["date"] >= "2026-06-28"]
    changed = True
    while changed:
        changed = False
        for fx in resolved_fixtures(bracket, known):
            if fx["id"] in known:
                continue
            w = _result_winner(ko, pens, fx["home"], fx["away"])
            if w:
                known[fx["id"]] = w
                changed = True
    return known


def resolved_fixtures(bracket, known):
    """Every fixture whose participants are determined: all of R16, later rounds once
    both feeder slots have real winners. Lets the pipeline advance round-by-round
    (cards, forecasts, pins) without hand-editing bracket_2026.json."""
    out = [dict(fx, round="r16") for fx in bracket["r16"]]
    for rnd in ("qf", "sf"):
        for fx in bracket[rnd]:
            if all(k in known for k in fx["from"]):
                out.append(dict(fx, round=rnd,
                                home=known[fx["from"][0]], away=known[fx["from"][1]]))
    f = bracket["final"]
    if all(k in known for k in f["from"]):
        out.append(dict(f, round="final", home=known[f["from"][0]], away=known[f["from"][1]]))
    tp = bracket.get("third_place")
    if tp and all(k in known for k in tp["from"]):
        losers = []
        for sf_id in tp["from"]:
            sf = next(s for s in bracket["sf"] if s["id"] == sf_id)
            if not all(k in known for k in sf["from"]):
                losers = None
                break
            parts = {known[sf["from"][0]], known[sf["from"][1]]}
            losers.append((parts - {known[sf_id]}).pop())
        if losers:
            out.append(dict(tp, round="third_place", home=losers[0], away=losers[1]))
    return out


def run_tournament(sim, bracket, known, n_sims, sampler="antithetic", return_paths=False):
    """Returns {team: [p_reach_QF, p_reach_SF, p_reach_Final, p_champion, p_bronze]};
    with return_paths also ({"teams": [...], "winners": {slot: int16 array}}) per sim."""
    teams = list(dict.fromkeys(
        [t for fx in bracket["r16"] for t in (fx["home"], fx["away"])] + list(known.values())
    ))
    team_to_i = {t: i for i, t in enumerate(teams)}
    reach = np.zeros((len(teams), 5), dtype=np.int64)
    draws = _uniforms(sim.rng, n_sims, 16, sampler)
    draw_col = 0
    winners, parts = {}, {}

    def ids(team):
        return np.full(n_sims, team_to_i[team], dtype=np.int16)

    def choose(fx, a, b):
        nonlocal draw_col
        if fx["id"] in known:
            return ids(known[fx["id"]])
        u = draws[:, draw_col]
        draw_col += 1
        out = np.empty(n_sims, dtype=np.int16)
        pairs = np.stack([a, b], axis=1)
        unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
        for pair_idx, (ai, bi) in enumerate(unique_pairs):
            mask = inverse == pair_idx
            p = sim.advance_prob(teams[int(ai)], teams[int(bi)], fx["venue_country"])
            out[mask] = np.where(u[mask] < p, ai, bi)
        return out

    rounds = [("r16", 0), ("qf", 1), ("sf", 2)]
    for rnd, level in rounds:
        for fx in bracket[rnd]:
            a, b = (ids(fx["home"]), ids(fx["away"])) if "home" in fx else (
                winners[fx["from"][0]], winners[fx["from"][1]]
            )
            parts[fx["id"]] = (a, b)
            winners[fx["id"]] = choose(fx, a, b)
            np.add.at(reach[:, level], winners[fx["id"]], 1)
    f = bracket["final"]
    winners[f["id"]] = choose(f, winners[f["from"][0]], winners[f["from"][1]])
    np.add.at(reach[:, 3], winners[f["id"]], 1)
    tp = bracket.get("third_place")
    if tp:
        losers = [np.where(winners[s] == parts[s][0], parts[s][1], parts[s][0])
                  for s in tp["from"]]
        winners[tp["id"]] = choose(tp, losers[0], losers[1])
        np.add.at(reach[:, 4], winners[tp["id"]], 1)
    probs = {teams[i]: row / n_sims for i, row in enumerate(reach) if row.sum()}
    if return_paths:
        return probs, {"teams": teams, "winners": winners}
    return probs


def run_ensemble(param_samples, pens, bracket, known, n_sims, sampler="antithetic",
                 seed=None, goal_scale=DEFAULT_GOAL_SCALE, external_strength=None,
                 external_weight=0.0, form_strength=None, form_weight=0.0,
                 live_strength=None, live_weight=0.0):
    """Mixture over bootstrap parameter samples (uncertainty.py): each sample simulates
    an equal share of paths, propagating estimation uncertainty into the bracket."""
    B = len(param_samples)
    per = [n_sims // B + (1 if i < n_sims % B else 0) for i in range(B)]
    rng = np.random.default_rng(seed)
    agg, all_winners, teams, total = {}, None, None, 0
    for ps, n in zip(param_samples, per):
        if n == 0:
            continue
        sim = Simulator(ps["attack"], ps["defence"], ps["hfa"], ps["rho"], rng,
                        pens=pens, goal_scale=goal_scale,
                        external_strength=external_strength, external_weight=external_weight,
                        form_strength=form_strength, form_weight=form_weight,
                        live_strength=live_strength, live_weight=live_weight)
        probs, paths = run_tournament(sim, bracket, known, n, sampler, return_paths=True)
        teams = paths["teams"]
        if all_winners is None:
            all_winners = {k: [v] for k, v in paths["winners"].items()}
        else:
            for k, v in paths["winners"].items():
                all_winners[k].append(v)
        for t, p in probs.items():
            agg[t] = agg.get(t, 0) + np.asarray(p) * n
        total += n
    winners = {k: np.concatenate(v) for k, v in all_winners.items()}
    return {t: p / total for t, p in agg.items()}, {"teams": teams, "winners": winners}


def verdict_bracket(sim, bracket, known):
    """Instant deterministic bracket: known facts plus highest advance-prob picks.

    This is separate from Monte Carlo champion/path consensus. It answers
    "forced pick right now" without conditioning every slot on one champion.
    """
    winners, losers, rows = dict(known), {}, []

    def add(fx, round_name, home, away):
        played = fx["id"] in known
        p_home = 1.0 if played and known[fx["id"]] == home else 0.0 if played else sim.advance_prob(
            home, away, fx["venue_country"])
        winner = known[fx["id"]] if played else home if p_home >= 0.5 else away
        support = 1.0 if played else max(p_home, 1.0 - p_home)
        loser = away if winner == home else home
        winners[fx["id"]] = winner
        losers[fx["id"]] = loser
        rows.append({
            "id": fx["id"], "round": round_name, "date": fx.get("date"),
            "home": home, "away": away, "winner": winner, "loser": loser,
            "played": played, "support": round(float(support), 4),
            "p_home_advance": round(float(p_home), 4),
            "p_away_advance": round(float(1.0 - p_home), 4),
        })

    for fx in bracket["r16"]:
        add(fx, "r16", fx["home"], fx["away"])
    for round_name in ("qf", "sf"):
        for fx in bracket[round_name]:
            add(fx, round_name, winners[fx["from"][0]], winners[fx["from"][1]])
    add(bracket["final"], "final",
        winners[bracket["final"]["from"][0]], winners[bracket["final"]["from"][1]])
    tp = bracket.get("third_place")
    if tp:
        add(tp, "third_place", losers[tp["from"][0]], losers[tp["from"][1]])
    final = next(r for r in rows if r["id"] == bracket["final"]["id"])
    return {
        "mode": "deterministic_advance_prob",
        "champion": final["winner"],
        "champion_match_support": final["support"],
        "matches": rows,
    }


# ---------------- reporting ----------------

def print_match_cards(sim, bracket, known):
    print("\n=== Round of 16 - Dixon-Coles match predictions ===")
    for fx in bracket["r16"]:
        a, b, venue = fx["home"], fx["away"], fx["venue_country"]
        lam, mu, _, g = sim.grid_for(a, b, venue)
        h, d, w, o25, top = markets(g)
        status = f"PLAYED — {known[fx['id']]} advanced" if fx["id"] in known else fx["date"]
        scores = "  ".join(f"{s} {p:.1%}" for p, s in top)
        print(f"\n{a} vs {b}  ({status}, {venue})")
        print(f"  lambda={lam:.2f} mu={mu:.2f} | {a} {h:.1%} / draw {d:.1%} / {b} {w:.1%} | O2.5 {o25:.1%}")
        print(f"  top scorelines: {scores}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    ap.add_argument("--sampler", choices=SAMPLERS, default="antithetic")
    ap.add_argument("--half-life", type=float, default=1100.0,
                    help="decay half-life, days (sweep-validated: smallest OOS gap)")
    ap.add_argument("--friendly-weight", type=float, default=1.0, help="weight multiplier for friendlies")
    ap.add_argument("--goal-scale", type=float, default=DEFAULT_GOAL_SCALE,
                    help="multiplicative score-rate calibration (walk-forward default: 1.10)")
    ap.add_argument("--external-weight", type=float, default=DEFAULT_EXTERNAL_WEIGHT,
                    help="capped player/market prior weight (0 disables)")
    ap.add_argument("--form-weight", type=float, default=DEFAULT_FORM_WEIGHT,
                    help="capped opponent-adjusted recent-form prior weight")
    ap.add_argument("--live-weight", type=float, default=DEFAULT_LIVE_WEIGHT,
                    help="capped current-tournament xG/stat momentum prior weight")
    ap.add_argument("--years", type=float, default=4.0, help="training window, years")
    ap.add_argument("--seed", type=int, default=26)
    args = ap.parse_args()

    df = load_matches(args.years)
    print(f"training on {len(df)} matches, {df['date'].min().date()} -> {df['date'].max().date()}, "
          f"{df['home_team'].nunique()} teams, half-life {args.half_life:.0f}d, "
          f"friendly weight {args.friendly_weight}")
    model = fit_model(df, args.half_life, args.friendly_weight)
    atk, dfn, hfa, rho = team_params(model)
    print(f"fitted: home_advantage={hfa:.3f} rho={rho:.3f}")

    bracket = json.loads((ROOT / "bracket_2026.json").read_text())
    shootouts = pd.read_csv(ROOT / "data" / "shootouts.csv", parse_dates=["date"])
    known = known_winners(bracket, df, shootouts)
    if known:
        print(f"known knockout results consumed: {known}")
    external_strength, external_meta = load_external_strength()
    if args.external_weight and external_strength:
        print(f"external strength prior: {external_meta['rows']} teams, weight {args.external_weight:.3f}")
    form_strength, form_meta = build_recent_form_strength(df, external_strength=external_strength)
    if args.form_weight and form_strength:
        print(f"recent-form prior: {form_meta['rows']} teams, weight {args.form_weight:.3f}")
    live_strength, live_meta = build_live_context_strength(
        df, features=load_match_features(ROOT), external_strength=external_strength)
    if args.live_weight and live_strength:
        print(f"live context prior: {live_meta['rows']} teams, weight {args.live_weight:.3f}")

    sim = Simulator(atk, dfn, hfa, rho, np.random.default_rng(args.seed),
                    pens=shootout_rates(shootouts), goal_scale=args.goal_scale,
                    external_strength=external_strength, external_weight=args.external_weight,
                    form_strength=form_strength, form_weight=args.form_weight,
                    live_strength=live_strength, live_weight=args.live_weight)
    print_match_cards(sim, bracket, known)

    probs = run_tournament(sim, bracket, known, args.sims, args.sampler)
    out = pd.DataFrame(
        [(t, *p) for t, p in probs.items()],
        columns=["team", "reach_QF", "reach_SF", "reach_final", "champion", "bronze"],
    ).sort_values("champion", ascending=False).reset_index(drop=True)

    print(f"\n=== Remaining-bracket Monte Carlo ({args.sims} sims, {args.sampler}) ===")
    print(out.to_string(index=False, formatters={c: "{:.1%}".format for c in out.columns[1:]}))

    (ROOT / "output").mkdir(exist_ok=True)
    out.to_csv(ROOT / "output" / "probabilities.csv", index=False)
    print("\nwrote output/probabilities.csv")


if __name__ == "__main__":
    main()
