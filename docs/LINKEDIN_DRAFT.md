# LinkedIn draft — edit voice/numbers before posting

> Numbers move every refresh. Before posting, pull fresh champion odds and
> forward-ledger stats from the UI. Model output ≠ betting advice.

---

⚽ I built a World Cup prediction engine while the World Cup was being played.

Not a retrospective. Not a toy trained once and screenshot-ed. A live engine
that re-scrapes results, refits its ratings, and re-simulates a million
tournament paths after every finished game — with every forecast recorded
*before* kickoff and scored publicly after.

The stack, in ~48 hours of building:

📊 Dixon-Coles (1997) time-weighted bivariate Poisson — attack/defence ratings
for 209 national teams, fit on ~3,800 internationals
🎲 1,000,000 bracket simulations per refresh, vectorized quasi-Monte-Carlo
(scrambled Sobol / Latin hypercube / antithetic)
🧮 16-refit bootstrap ensemble, because a point estimate that doesn't know
its own error bars is just confident noise
🥅 Proper knockout resolution: Poisson extra time, then historically-shrunk
penalty-shootout rates — not a coin flip
🔁 Self-updating scrapers (bulk dataset + same-day ESPN top-up), async
refresh, self-healing uncertainty ensemble

But here's the part I actually care about — validation:

✅ 394 out-of-sample matches, walk-forward: RPS 0.152 vs 0.236 random —
~36% better than chance, with a near-zero overfit gap (+0.007)
✅ Hyperparameters chosen by out-of-sample sweep, not vibes. Fun finding:
aggressive recency weighting looked BEST in-sample and WORST out-of-sample.
The overfit trap, caught and documented.
✅ A public forward ledger: forecasts recorded pre-match, scored post-match,
no edits. 5 of 6 knockout favorites hit so far.
❌ And one honest miss on the record: the model had Brazil over Norway.
Norway won. Worst forward score of the tournament. It stays on the ledger.

The uncomfortable lesson: writing the prediction model was the easy part.
Almost all the real work — and all the credibility — lives in the validation
loop: leak-proof forecast scoring, reliability bins, evidence logs, and the
discipline to reject features that couldn't prove themselves out-of-sample
(a recent-form prior is wired in and switched OFF, because the sweep said so).

Anyone can list twenty engine names in a feature list. The moat is a number:
how do you score on matches you've never seen?

Open source (AGPL-3.0), single-file vanilla-JS wallchart UI, evidence log
included: [repo link]

What's the most uncomfortable thing a backtest has ever told you?

#Python #DataScience #SportsAnalytics #FootballAnalytics #MachineLearning
#MonteCarlo #WorldCup2026 #OpenSource

---

## Alternate hooks (pick one)

1. "My model just lost money on Norway. I'm posting it anyway." (miss-first
   angle; strongest engagement, most credible)
2. "394 out-of-sample matches. RPS 0.152 vs 0.236 random. That one number
   took more work than the entire model." (metric-first)
3. "It'll probably take a weekend." — it did, but only because I refused to
   write an MLE solver that already exists. (OSS-leverage angle; nods to the
   post that inspired the comparison)

## Facts checklist (verify before posting)

- [ ] Champion odds current (UI header timestamp fresh)
- [ ] Forward ledger counts (settled / favorites hit)
- [ ] RPS + match count from /api/backtest
- [ ] Repo link + license badge render on GitHub
- [ ] Disclaimer line present
