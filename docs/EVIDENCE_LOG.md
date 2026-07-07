# Evidence Log

Last updated: 2026-07-05

## Bias Diagnostics

Command:

```powershell
.venv\Scripts\python.exe diagnostics.py --start 2026-01-01 --refit-days 45 --half-life 550 --friendly-weight 1.0
```

Result:

- Scored matches: 391
- Skipped: 0
- RPS: 0.1578
- Confederation coverage: 210/210 teams
- Favorite predicted: 0.5747
- Favorite observed: 0.5985
- Draw predicted: 0.2487
- Draw observed: 0.2506

Largest exact-scoreline residuals:

| Score | Predicted | Observed | Gap |
| --- | ---: | ---: | ---: |
| 0-0 | 0.1047 | 0.0742 | -0.0305 |
| 2-2 | 0.0317 | 0.0537 | +0.0220 |
| 1-0 | 0.1109 | 0.0921 | -0.0188 |
| 2-0 | 0.0877 | 0.0691 | -0.0186 |
| 1-1 | 0.1063 | 0.1202 | +0.0139 |

Evidence does not support the original hypothesis that most match probabilities
are compressed between 0 and 1 because of scoreline amplification. The largest
scoreline issue is 0-0 overprediction, not a universal 0/1 amplification bug.

## Repair Checks

Friendly downweight candidates were tested before changing defaults.

| Half-life | Friendly weight | RPS | Brier | Log-loss | In-sample RPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 550 | 1.0 | 0.1578 | 0.4968 | 0.8509 | 0.1427 |
| 550 | 0.6 | 0.1586 | 0.4983 | 0.8535 | 0.1441 |
| 550 | 0.3 | 0.1598 | 0.5010 | 0.8577 | 0.1460 |

Decision: keep `friendly_weight=1.0`. The proposed repair worsened
out-of-sample RPS.

## Simulation Methods

The bracket simulator now uses 1,000,000 vectorized paths by default with
antithetic uniforms. LHS and Sobol samplers are available. Copulas,
GBM/mean-reversion, importance sampling, multiprocessing, and GPU acceleration
remain off because there is no validated correlated macro factor path or
rare-tail payoff target in the current football model.

## Match Features

Command:

```powershell
.venv\Scripts\python.exe match_features.py --start 2026-06-01
```

Result:

- ESPN feature rows: 88
- xG pair coverage: 79.5%
- shots/SOT/corners/possession/fouls coverage: 100%
- Diagnostics joined scored rows: 54
- Diagnostics joined xG rows: 45
- Average joined goal total: 3.0444
- Average joined xG total: 2.7030
- xG/result disagreement: 35.56%

Decision: use match-wise stats and xG only as post-match diagnostic evidence
for now. They are not forecast inputs because using current-match xG before
kickoff would leak the outcome; rolling xG calibration should wait for enough
settled, no-leak history.

## Forward Calibration — Canada vs Morocco (2026-07-04)

First knockout fixture settled by the forward ledger (no leakage: forecast
recorded pre-match, scored post-match):

- Result: Canada 0-3 Morocco. Model favorite: Morocco at 45.1% -> favorite hit.
- Forward RPS 0.1669 (backtest average 0.1578; single-match noise applies).
- The result entered training data same-day via the ESPN top-up scraper and the
  model refit consumed it (`known` now pins R16-1 = Morocco).

Loop bug found and fixed while settling: the ledger records one forecast per
refresh, so the same fixture was being scored 13 times, overweighting
re-forecast fixtures. `settle_forward_forecasts` now scores only the latest
pre-match forecast per fixture (`revisions_excluded` reported; history kept).

## Parameter Uncertainty — Bootstrap Ensemble (2026-07-04)

`uncertainty.py` resamples matches with replacement and refits the MLE B times;
`wc_sim.run_ensemble` mixes simulation paths across samples. Addresses the
overconfidence blindspot (point estimates ignore estimation error). The server
uses `output/param_samples.json` only when it matches current knobs and data
date, else falls back to the point estimate — stale uncertainty is worse than
none.

## Queue Closures (2026-07-04)

- xG-blended ratings: PARKED with evidence. Only 89 WC matches carry xG, the
  penaltyblog likelihood requires integer goals, and the forward ledger has one
  settled knockout fixture — any blend weight would be unvalidatable. Data
  ingestion stays live (`match_features.py`); revisit when the settled forward
  sample is meaningful.
- Market anchor: PARKED. No keyless reliable WC-2026 odds source; the-odds-api
  requires a signup key. Wire-up documented in README roadmap; needs owner key.
- Player/lineup layer: PARKED. EA-attribute engines shown in the referenced
  LinkedIn build ship no calibration evidence; adopting their inputs without
  validation would regress the pipeline's evidence discipline.

## License (2026-07-04)

MIT -> AGPL-3.0. The engine is deployed as a network service (Flask app);
AGPL's network-copyleft keeps hosted derivatives open. All runtime deps are
permissive (MIT/BSD) and AGPL-compatible; martj42 data is CC0; sole author
relicense.

## Bracket Display and Goal-Scale Repair (2026-07-05)

Bug: the wallchart showed champion-anchored conditional support as if it were
individual bracket odds. Example: after choosing Argentina as modal champion,
the coherent path could show Morocco over France at 100.0% because that QF was
already implied inside the path subset. Fix: `consensus.py` now emits `slot_p`
for unconditional slot probability; `web/index.html` displays `slot_p` and
keeps conditional support only in hover/details. The definitive champion remains
visible.

Low-score calibration check:

| Half-life | Goal scale | RPS | Brier | Log-loss | 0-0 gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1100 | 1.00 | 0.1578 | 0.4961 | 0.8503 | not current |
| 1100 | 1.10 | 0.1571 | 0.4949 | 0.8493 | -0.011 |

Decision: use `goal_scale=1.10` by default. It improves out-of-sample RPS,
Brier, and log-loss, and reduces the observed 0-0 overprediction without
claiming player/lineup signals the model does not yet validate.

## External Player and Market Mart (2026-07-05)

Command:

```powershell
.venv\Scripts\python.exe external_data.py
```

Result:

- Sources: `dcaribou/transfermarkt-datasets` plus FIFA world rankings override
  from `https://www.fifa.com/en/world-rankings`
- Generated compact ignored outputs under `output/external/`
- `team_strength.csv`: 124 Transfermarkt national-team rows before rank-only
  fallbacks
- `player_pool.csv`: 2,852 top-23 players across national teams
- `team_chemistry.csv`: 124 national teams
- `project_team_enrichment.csv`: 124 Transfermarkt national-team rows before
  rank-only fallbacks
- Player/market fields now visible in `/api/data` and the web UI:
  FIFA rank, current national-team player count, top-11 market value,
  top-23 market value, squad caps, squad goals, chemistry score, position
  balance, and same-club share

Decision update: feed external context into the model as a capped prior, not as
a hard override. `external_signals.py` builds a normalized composite from
top-23 market value, current FIFA rank, caps/goals, and chemistry. The prior
adjusts both teams' goal rates symmetrically, capped by `MAX_RATE_ADJ=0.25`.

Quick OOS check after rank/chemistry integration (`half_life=1100`,
`goal_scale=1.10`):

| External weight | RPS | Brier | Log-loss | OOS gap |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.1571 | 0.4949 | 0.8493 | +0.0097 |
| 0.06 | 0.1551 | 0.4910 | 0.8427 | +0.0087 |
| 0.10 | 0.1540 | 0.4890 | 0.8394 | +0.0080 |
| 0.12 | 0.1537 | 0.4884 | 0.8385 | +0.0078 |
| 0.15 | 0.1532 | 0.4876 | 0.8375 | +0.0073 |

Default: `external_weight=0.12`. The tested cap `0.15` scored best, but using
a default below the cap keeps the integration from silently becoming a
market/ranking model.

Update: teams present in FIFA rankings but absent from the Transfermarkt
national-team roster are now added as rank-only fallback rows. Cape Verde is
the first live example: FIFA's Cabo Verde detail page reports rank 67 on the
11 June 2026 update, while the Transfermarkt national-team table has no Cape
Verde/Cabo Verde row. The mart therefore keeps Cape Verde's player, market,
caps, goals, and chemistry fields null, and `external_signals.py` gives missing
components neutral zero z-score contribution instead of imputing a fake average
squad. Regenerated rows: `team_strength=125`,
`fifa_only_team_strength=1`, `project_team_enrichment=125`.

Backtest after fallback remains stable:

- RPS `0.1537`
- Brier `0.4884`
- Log-loss `0.8385`
- In-sample RPS `0.1460`, OOS gap `+0.0077`

## Recent-form prior bias check

The Cape Verde/Canada complaint is not fully explained by missing FIFA rank:
with the rank-only fallback, Cape Verde still rates as an underdog. I added a
capped `form_signals.py` prior that scores recent results against rank/market
expectations and can use prior-match xG when present. It is wired through
`match_rates`, the server, the UI, and backtest, but defaults to `0.00` because
the walk-forward sweep showed overfit:

| Form weight | RPS | Brier | Log-loss | In-sample RPS | OOS gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.1537 | 0.4884 | 0.8385 | 0.1460 | +0.0077 |
| 0.005 | 0.1538 | 0.4886 | 0.8388 | 0.1449 | +0.0089 |
| 0.010 | 0.1539 | 0.4888 | 0.8392 | 0.1438 | +0.0101 |
| 0.020 | 0.1542 | 0.4895 | 0.8402 | 0.1417 | +0.0125 |
| 0.030 | 0.1545 | 0.4902 | 0.8414 | 0.1398 | +0.0147 |
| 0.040 | 0.1549 | 0.4911 | 0.8427 | 0.1379 | +0.0170 |

Decision: keep the form prior as an inspectable/tunable engine and show it in
`/api/case`, but leave the default at `0.00` until more settled forward data
proves that it improves forecasts without widening the in-sample/OOS gap.

Update: the form engine now also uses prior-match shot pressure (shots, shots
on target, corners, possession) when ESPN features exist. The signal remains
forward-safe because only matches before `as_of` are consumed. Retest after the
stat-pressure addition still does not justify enabling it by default:

| Form weight | RPS | Brier | Log-loss | In-sample RPS | OOS gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.1526 | 0.4857 | 0.8348 | 0.1453 | +0.0073 |
| 0.003 | 0.1527 | 0.4859 | 0.8350 | 0.1447 | +0.0080 |
| 0.005 | 0.1528 | 0.4860 | 0.8352 | 0.1442 | +0.0086 |
| 0.010 | 0.1529 | 0.4863 | 0.8357 | 0.1432 | +0.0097 |
| 0.020 | 0.1532 | 0.4870 | 0.8368 | 0.1411 | +0.0121 |
| 0.040 | 0.1541 | 0.4889 | 0.8397 | 0.1373 | +0.0168 |

## Full live FIFA ranking sync (2026-07-05)

Command:

```powershell
.venv\Scripts\python.exe fifa_rankings.py
.venv\Scripts\python.exe external_data.py
.venv\Scripts\python.exe backtest.py
```

Result:

- FIFA live endpoint source: `https://www.fifa.com/en/world-rankings`
- Synced rows: 211 men's teams
- Live top five: France, Argentina, Spain, England, Brazil
- Cape Verde alias: FIFA `Cabo Verde` -> model `Cape Verde`; live rank `64`
- External mart rows: `project_team_enrichment=216`, `fifa_only_team_strength=92`
- Name aliases now bridge common model/data mismatches: USA, Cabo Verde,
  IR Iran, Korea Republic, Korea, South, Korea, North, DPR Korea, China PR,
  Czechia, Ireland, Türkiye.

Backtest after full-rank integration (`refit_days=45`, `half_life=1100`,
`goal_scale=1.10`, `external_weight=0.12`, `form_weight=0.00`):

- RPS `0.1525`
- Brier `0.4855`
- Log-loss `0.8345`
- In-sample RPS `0.1453`, OOS gap `+0.0072`

Decision: keep `external_weight=0.12`. Full live ranks improve OOS metrics
without increasing the overfit gap, while rank-only teams still keep missing
player/market/chemistry fields null instead of imputing fake squads.

## Forward calibration loop

The forward ledger is now explicitly self-feeding but sample-gated. Each
refresh records current unplayed fixture forecasts, settles only the latest
pre-match forecast per fixture, and writes a `calibration_policy` into
`output/forward_calibration.json` plus `meta.forward_loop`. The policy holds
until at least 12 settled pre-match forecasts exist. Once that threshold is
reached it uses favorite predicted-vs-observed gap to recommend one of:

- `reduce_prior_or_goal_confidence`
- `allow_slightly_more_prior_confidence`
- `keep_current_defaults`

Current live state: one settled forecast, so policy is `hold`. This prevents a
single Morocco/Canada hit or miss from auto-tuning the model.

## Pipeline Upgrades (2026-07-05)

- Bracket auto-advance: `wc_sim.resolved_fixtures` materializes any slot whose
  feeder slots have real winners (QF-1 = Morocco vs France appeared the moment
  both R16 results landed). Cards, forward-ledger forecasts, and what-if pins
  now follow the tournament round-by-round without editing bracket_2026.json.
- Async refresh: POST /api/refresh returns immediately; a background job runs
  scrape -> refit -> simulate and reports progress via GET /api/status. If the
  bootstrap ensemble is stale for the new data date, the job regenerates 16
  resample refits and re-simulates, so the ensemble no longer silently degrades
  to a point estimate after each matchday.
- What-if pinning extended to every determined unplayed slot; played slots are
  explicitly not re-pinnable by default (real results are facts).
- Separate counterfactual mode now allows played-slot rewrites, e.g. "what if
  Canada had beaten Morocco", and drops downstream known facts that depended on
  the real result before re-simulating.
- Cached server state now rejects stale no-bronze payloads on startup, forcing a
  rebuild when the bracket schema changes instead of serving old JSON.
- The web UI exposes bronze/third-place odds when present and preserves
  recomputed what-if pins in the URL hash for reopening a scenario.
- /api/sample draws parameters from a random bootstrap sample when the ensemble
  is active (scoreline noise + estimation noise).
- Verified live: async job 67s to idle; QF-1 pin gives coherent conditional
  odds (Morocco reach-SF -> 0 under a France pin); invalid pins rejected.

## Instant deterministic verdict bracket (2026-07-05)

Problem: with roughly 10 tournament days left, the product needs pre-match
decisions, not a model that only becomes confident after facts arrive. The
existing champion-anchored consensus was coherent but could make individual
slot support look like a conditional artifact.

Change:

- Added `wc_sim.verdict_bracket`: a deterministic forced-pick path built from
  known facts plus each fixture's knockout `advance_prob` (90 minutes + extra
  time + penalties), not from champion-conditioned Monte Carlo paths.
- Added `payload.verdict` and what-if `verdict` responses.
- The web static bracket now renders that verdict and shows the forced-pick
  champion separately from championship probability tables.
- Cached state now rejects payloads without `verdict`, forcing one refresh when
  the schema changes.

Validation:

```powershell
.venv\Scripts\python.exe test_wc_sim.py
.venv\Scripts\python.exe -m py_compile wc_sim.py server.py test_wc_sim.py
git diff --check
```

Result: all self-checks passed, including a new guard that played slots are
100% only as facts while unplayed deterministic support comes from normal
match-level advance probabilities.

## Deeper external prior integration (2026-07-05)

User concern: enriched player/market/chemistry data must affect the model, not
only the UI. The previous prior used top-23 market value, FIFA rank, caps,
goals, and chemistry. I tested deeper composites against the same walk-forward
blocks, fitting each block once and rescoring candidate strength maps.

Prototype result (`392` OOS matches, `half_life=1100`, `goal_scale=1.10`,
`form_weight=0.00`):

| Composite | Weight | RPS | Brier | Log-loss |
| --- | ---: | ---: | ---: | ---: |
| rank_quality | 0.15 | 0.1512 | 0.4828 | 0.8316 |
| deep | 0.15 | 0.1517 | 0.4840 | 0.8329 |
| current | 0.15 | 0.1520 | 0.4845 | 0.8331 |
| current default before change | 0.12 | 0.1526 | 0.4857 | 0.8348 |

Accepted change: `external_signals.py` now uses top-11 quality, top-23 depth
beyond the top 11, FIFA rank, caps, goals, chemistry, position balance, and
same-club share. Missing fields remain neutral, preserving rank-only fallback
teams without fake squad imputation.

Full backtest after implementation:

- RPS `0.1512`
- Brier `0.4828`
- Log-loss `0.8316`
- In-sample RPS `0.1446`, OOS gap `+0.0066`

Decision: set default `external_weight=0.15`. This is the tested cap, but the
rate adjustment itself remains capped by `MAX_RATE_ADJ=0.25`, and the form
prior stays disabled because its previous sweep widened the overfit gap.

## Low-score modal scoreline diagnosis (2026-07-05)

User concern: match cards still often show 1-0, 0-0, 0-1, and 1-1 as top
scorelines. I checked whether this was an aggregate under-goals problem before
touching the model.

Goal-scale sweep on the same fitted walk-forward blocks:

| Goal scale | RPS | Brier | Log-loss | Pred goals | Actual goals | Pred O2.5 | Actual O2.5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.95 | 0.1518 | 0.4833 | 0.8302 | 2.547 | 2.804 | 0.447 | 0.518 |
| 1.00 | 0.1515 | 0.4828 | 0.8301 | 2.681 | 2.804 | 0.477 | 0.518 |
| 1.05 | 0.1513 | 0.4827 | 0.8306 | 2.815 | 2.804 | 0.505 | 0.518 |
| 1.10 | 0.1512 | 0.4828 | 0.8316 | 2.949 | 2.804 | 0.533 | 0.518 |
| 1.15 | 0.1512 | 0.4832 | 0.8331 | 3.083 | 2.804 | 0.559 | 0.518 |
| 1.20 | 0.1512 | 0.4838 | 0.8349 | 3.217 | 2.804 | 0.585 | 0.518 |

Finding: default `goal_scale=1.10` is not under-goal; it slightly overpredicts
total goals and over-2.5. Raising rates to make cards look less conservative
would be cosmetic and less calibrated.

Change:

- `backtest.py` now writes `scoreline_calibration`: predicted/actual goals,
  predicted/actual over-2.5, exact-score log-loss, top-1/top-3 exact-score hit
  rate, and share of matches whose modal predicted scoreline is low-score
  (`<=2` total goals).
- The validation UI renders these metrics.

Current full backtest scoreline metrics:

- Predicted goals `2.949` vs actual `2.804`
- Predicted over-2.5 `0.533` vs actual `0.518`
- Exact-score top-1 hit `0.166`, top-3 hit `0.357`
- Low-score modal top pick share `0.801`

Decision: keep `goal_scale=1.10` for now. The remaining issue is distribution
shape / exact-score concentration, not aggregate scoring level.

## Forward calibration self-feed guard (2026-07-05)

Before this change the forward loop recorded forecasts and wrote a
`calibration_policy`, but refreshes only displayed the policy. It did not feed
any calibration back into future predictions.

Change:

- `server._apply_forward_calibration()` reads the previous
  `output/forward_calibration.json` at refresh start.
- If policy is still `hold`, `keep_current_defaults`, or missing, it records no
  model change.
- Once the forward loop has enough settled pre-match forecasts and emits
  `reduce_prior_or_goal_confidence` or `allow_slightly_more_prior_confidence`,
  refresh applies a single bounded `external_weight` nudge of `-0.01` or
  `+0.01`.
- `output/forward_calibration_applied.json` stores the applied report id, so
  the same report cannot ratchet the weight on every auto-refresh.
- `meta.forward_calibration_applied` and the header expose whether a nudge was
  applied.

Current live policy remains `hold` because only `2` settled pre-match forecasts
exist versus the `12` minimum. That is intentional: the feedback loop is wired,
but not allowed to tune from two matches.

Validation:

```powershell
.venv\Scripts\python.exe test_wc_sim.py
.venv\Scripts\python.exe -m py_compile server.py test_wc_sim.py forward_loop.py
git diff --check
```

Result: all self-checks passed, including hold/no-op and apply-once/idempotence
coverage for the calibration hook.

## Player usage coverage refresh (2026-07-05)

The previous generated external mart had `include_usage=false`, so the
Transfermarkt player/market layer was present but live World Cup player usage
columns were all zero. I regenerated with usage enabled and made usage mode the
default for `external_data.py` (`--skip-usage` is now the fast opt-out).

Command:

```powershell
.venv\Scripts\python.exe external_data.py --include-usage
```

Result:

- `project_team_enrichment=215`
- `player_pool=2852`
- `team_chemistry=124`
- `fiwc_2026_appearances=44`
- `fiwc_2026_starts=0`

Coverage details:

- 43 project teams have nonzero `fiwc_minutes` after usage aliasing.
- Current bracket examples now have usage minutes/goals/assists in
  `/api/data`: Canada, Morocco, France, Paraguay, United States, Mexico,
  Argentina, Brazil, Norway.
- FIWC starts remain unavailable because the upstream `game_lineups` table has
  zero rows matching FIWC games after 2026-06-01, despite the `games` table
  containing FIWC fixtures.

Change:

- `external_data.py` now aliases usage rows through the same team alias table
  used by team strength.
- `external_data.py` defaults to include usage; `--skip-usage` keeps the faster
  rebuild path.
- `/api/data` external payload now keeps appearances, minutes, goals, assists,
  cards, and start fields.
- The web player/market layer shows usage coverage summary and lineups row
  count, so the missing-starter limitation is visible instead of hidden.

Decision: do not create a missing-starter model prior yet. There are no FIWC
lineup rows to validate or apply it. Current usage is useful context, but
starter-specific adjustments would be fake precision.

## Active-star x-factor guard (2026-07-07)

User concern: market-only top-player labels can over-credit expensive players
who are not currently starting or producing, while active stars such as Haaland
should be visible in the model without team-specific exceptions.

Decision: keep the static market-star component, but reduce its weight inside
the live external prior and add an active-star component from
`fiwc_top_market_usage_share` and `fiwc_top_market_impact_score`. Missing usage
stays neutral. The change rewards used/scoring stars and reduces the live edge
from quiet/benched market stars without hard-coding player names.

## Availability cache invalidation (2026-07-07)

User concern: late injuries/suspensions matter, but automatic lineup feeds are
not reliable enough for unvalidated missing-starter adjustments.

Decision: keep the manual `data/availability.json` layer, but make it part of
the state cache key. `server.py` now stores a SHA-256 fingerprint of the
availability file in `meta.availability_input`; startup refreshes the bracket
when that file changes, so manual injury/suspension updates cannot silently
decorate stale odds.

Update: generalized the cache key into `meta.model_input_signature`. The server
now fingerprints `matches.csv`, `shootouts.csv`, `match_features.csv`,
`project_team_enrichment.csv`, `availability.json`, and `param_samples.json`.
Any changed model input forces startup refresh before odds are treated as
current. `load_state()` also reloads matching bootstrap samples, so what-if
simulations keep ensemble uncertainty after restart instead of falling back to
a point estimate.

## Live-context overfit repair (2026-07-07)

User concern: form, momentum, xG, and star context should feed the model, but
not turn one match into a hard narrative. Repair: current-tournament live
strength still uses result residuals, xG, and stat pressure, then shrinks each
team's live z-score by completed-match confidence up to three matches. One-match
signals now move odds directionally without receiving full live-context weight.

## Scoreline dispersion repair (2026-07-07)

User concern: exact-score cards were too concentrated around 0-0, 1-0, 0-1,
and 1-1. Raising `goal_scale` would inflate total goals, so I added a small
scoreline-only tempo mixture: each grid averages low/normal/high match tempo
with weights 25/50/25. This reduces exact-score overconfidence while preserving
the fitted mean rates.

Walk-forward check (`396` OOS matches, `half_life=1100`, `goal_scale=1.10`,
`external_weight=0.15`, `form_weight=0.00`):

| Score spread | RPS | Brier | Log-loss | Exact-score log-loss | Pred goals | Pred O2.5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.1516 | 0.4825 | 0.8311 | 2.8903 | 2.951 | 0.534 |
| 0.05 | 0.1516 | 0.4825 | 0.8311 | 2.8902 | 2.951 | 0.533 |
| 0.10 | 0.1516 | 0.4825 | 0.8309 | 2.8899 | 2.951 | 0.532 |
| 0.15 | 0.1516 | 0.4824 | 0.8307 | 2.8896 | 2.951 | 0.531 |

Decision: default `scoreline_dispersion=0.10`. It moves exact-score sharpness
in the right direction without changing the 1X2 metrics or aggregate goal
level materially.

## Backlog Closure (2026-07-06)

- Market anchor built (`market_anchor.py`, `/api/market`, UI section): de-vigs
  each bookmaker, averages implied champion odds, 50/50 log-pool blend shown as
  a benchmark. Deliberately NOT a model input. Degrades to a clear note without
  THE_ODDS_API_KEY. Devig/log-pool math unit-tested.
- Manual availability layer (`availability.py` + `data/availability.json`):
  missing-player value share (capped 0.5/team) subtracts from external strength,
  riding the capped external-prior channel (weight <= 0.15). No file = no-op.
  Unknown teams reported, not applied. Unit-tested.
- Everything else from the backlog already landed earlier: bronze match sim,
  counterfactual played-slot rewrites, shareable what-if URLs, gzip responses,
  WC26_TOKEN bearer auth on mutating endpoints.
