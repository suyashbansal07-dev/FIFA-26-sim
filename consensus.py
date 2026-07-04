"""Consensus champion + coherent path across the Monte Carlo bracket ensemble.

Two empirical views over the same simulated paths:
  - top_paths: joint mode — the most frequent complete tournament outcomes
  - consensus_path: champion-anchored conditional mode — pick the modal champion,
    then resolve every remaining slot (final -> SF -> QF -> R16) by conditional
    frequency among the sims still consistent with all picks so far. Coherent by
    construction: every pick is drawn from a non-empty consistent subset.
"""
from __future__ import annotations

import numpy as np


def slot_order(bracket):
    """Reverse round order: final -> SF -> QF -> R16 (champion-anchored resolution)."""
    return ([bracket["final"]["id"]] + [f["id"] for f in bracket["sf"]]
            + [f["id"] for f in bracket["qf"]] + [f["id"] for f in bracket["r16"]])


def build_consensus(paths, bracket, known, top_k=5):
    teams, winners = paths["teams"], paths["winners"]
    slots = slot_order(bracket)
    n = len(next(iter(winners.values())))
    base = len(teams)
    mat = np.stack([np.asarray(winners[s], dtype=np.int64) for s in slots], axis=1)

    # joint mode: pack each path into one integer, count uniques
    assert base ** mat.shape[1] < 2 ** 63, "path packing overflow"
    packed = np.zeros(n, dtype=np.int64)
    for c in range(mat.shape[1]):
        packed = packed * base + mat[:, c]
    vals, counts = np.unique(packed, return_counts=True)
    order = np.argsort(counts)[::-1][:top_k]
    top_paths = []
    for v, cnt in zip(vals[order], counts[order]):
        path, rem = {}, int(v)
        for s in reversed(slots):
            rem, idx = divmod(rem, base)
            path[s] = teams[idx]
        top_paths.append({"path": path, "count": int(cnt), "share": round(cnt / n, 5)})

    # champion-anchored conditional mode
    consistent = np.ones(n, dtype=bool)
    picks = []
    for col, slot in enumerate(slots):
        sub = mat[consistent, col]
        idx = int(np.bincount(sub, minlength=base).argmax())
        p = float((sub == idx).mean())
        consistent &= mat[:, col] == idx
        picks.append({"slot": slot, "winner": teams[idx], "conditional_p": round(p, 4),
                      "support": int(consistent.sum()), "known": slot in known})
    return {
        "sims": n,
        "modal_champion": {"team": picks[0]["winner"], "p": picks[0]["conditional_p"]},
        "consensus_path": {"picks": picks,
                           "joint_support": round(int(consistent.sum()) / n, 6)},
        "top_paths": top_paths,
    }
