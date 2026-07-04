# 2026 World Cup — Dixon-Coles Simulator

Time-weighted Dixon-Coles (1997) engine per `Dixon-Coles Architecture Spec.md`,
with a web UI, scrapers that pick up every finished game, and a walk-forward
calibration backtest. Fits attack/defence/home-advantage/rho on ~4 years of
international results (exponential decay, neutral-venue aware), builds
tau-corrected scoreline grids, Monte-Carlo simulates the remaining bracket.

## Run

```powershell
py -3.13 -m venv .venv                      # penaltyblog has no cp314 wheels yet
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python server.py              # http://127.0.0.1:8026
```

First boot scrapes, fits, and simulates automatically (~15 s). The UI has a
Refresh button; the server also auto-refreshes every 6 h (`--auto-refresh-hours`).
CLI equivalents: `fetch_data.py`, `wc_sim.py --sims 10000`, `backtest.py`,
`test_wc_sim.py`.

Fetched CSVs in `data/` and generated files in `output/` are intentionally
ignored by git; rerun the scripts above to rebuild them.

## Pieces

- `server.py` — Flask: web UI, `/api/data`, `/api/predict?home=X&away=Y[&venue=C]`,
  `/api/sample` (draws resolve to pens), `POST /api/refresh` (scrape → refit → re-sim,
  ~13 s), `GET/POST /api/backtest`, auto-refresh loop, state persisted to `output/state.json`
- `web/index.html` — vanilla single-file UI: championship odds, R16 match cards,
  any-matchup predictor with scoreline heatmap, sample-a-result, team ratings
- `fetch_data.py` — scrapers: martj42/international_results bulk + ESPN scoreboard
  same-day top-up (finished games incl. shootout winners; dedup across UTC skew)
- `wc_sim.py` — model core: penaltyblog MLE fit (`neutral_venue`-aware), grids,
  Monte Carlo, CLI report; writes `output/probabilities.csv`. Training pool is
  FIFA-competition teams only (drops CONIFA/regional sides the dataset carries)
- `backtest.py` — walk-forward calibration: monthly refits, out-of-sample RPS /
  Brier / log-loss vs uniform + train-frequency baselines.
  Latest: **RPS 0.1578 vs 0.236 uniform** over 391 matches (Jan–Jul 2026)
- `bracket_2026.json` — remaining bracket state (pairings, QF/SF tree, venues);
  played knockout winners are consumed automatically from the data
- Knobs: `--sims`, `--half-life` (550 d default), `--years` (4), `--seed`

## Known limitations (spec §5 + deliberate cuts)

- No lineup/injury awareness; ratings move only through results.
- Knockout draws use simple proportional extra-time goals, then Beta-shrunk
  historical shootout rates.
- No confederation multiplier; 4 years of friendlies+qualifiers connect the graph.
- Group-stage/best-thirds simulation not implemented (tournament already past it).

## License

MIT.
