"""Parameter-uncertainty propagation: weighted bootstrap refits of the Dixon-Coles MLE.

Point-estimate ratings ignore estimation error — the sim is too sure of itself
(the "squad-strength variance / X-factor" blindspot). Resampling matches with
replacement and refitting B times yields an empirical parameter distribution;
the bracket sim then mixes simulation paths across samples (wc_sim.run_ensemble),
widening tournament tails honestly.

Run: .venv/Scripts/python uncertainty.py [--boots 16] [--half-life 550]
     [--friendly-weight 1.0] [--years 4] [--seed 26]
Writes output/param_samples.json; the server uses it automatically when it matches
the current knobs and data date, else falls back to the point estimate.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wc_sim import ROOT, fit_model, load_matches, team_params

OUT = ROOT / "output"


def bootstrap_samples(df, boots, half_life, friendly_weight, seed=26, required=()):
    rng = np.random.default_rng(seed)
    samples = []
    while len(samples) < boots:
        boot = df.iloc[rng.integers(0, len(df), len(df))].reset_index(drop=True)
        atk, dfn, hfa, rho = team_params(fit_model(boot, half_life, friendly_weight))
        if any(t not in atk for t in required):
            continue  # redraw: a required team fell out of the resample
        samples.append({"attack": atk, "defence": dfn, "hfa": hfa, "rho": rho})
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boots", type=int, default=16)
    ap.add_argument("--half-life", type=float, default=550.0)
    ap.add_argument("--friendly-weight", type=float, default=1.0)
    ap.add_argument("--years", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=26)
    args = ap.parse_args()

    df = load_matches(args.years)
    bracket = json.loads((ROOT / "bracket_2026.json").read_text())
    alive = sorted({t for fx in bracket["r16"] for t in (fx["home"], fx["away"])})
    print(f"bootstrapping {args.boots} refits on {len(df)} matches "
          f"(half-life {args.half_life:.0f}d, friendly weight {args.friendly_weight})...")
    samples = bootstrap_samples(df, args.boots, args.half_life, args.friendly_weight,
                                args.seed, required=alive)

    OUT.mkdir(exist_ok=True)
    (OUT / "param_samples.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_max_date": str(df["date"].max().date()),
        "boots": args.boots, "half_life": args.half_life,
        "friendly_weight": args.friendly_weight, "samples": samples,
    }))

    print("attack-rating dispersion across refits (alive teams):")
    for t in alive:
        vals = np.array([s["attack"][t] for s in samples])
        print(f"  {t:15} mean {vals.mean():+.3f}  std {vals.std():.3f}")
    print(f"\nwrote output/param_samples.json ({args.boots} samples)")


if __name__ == "__main__":
    main()
