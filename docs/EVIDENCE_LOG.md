# Evidence Log

Last updated: 2026-07-04

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
