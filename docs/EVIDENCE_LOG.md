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
