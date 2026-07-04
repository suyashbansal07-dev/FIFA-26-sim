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
