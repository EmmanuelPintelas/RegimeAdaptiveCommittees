"""Overfitting of greedy subset search with pure-noise members (Proposition 17).

M members, each right on each tuning point independently with the same probability
a, two classes, plurality vote with ties counted as errors.  Every committee of size
k then has the same true accuracy, P[Bin(k, a) > k/2], so the tuning-split accuracy
of the greedy prefix of size k minus that number is exactly how much the search has
overfit.  It is compared with the deviation of Proposition 17 at k_max = k and with
the k_max = 1 lower bound of its part (ii).  Writes results/main/pure_noise.json.

    python -m analysis.pure_noise
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy.stats import binom

from .run import OUT

M = 100
ACC = 0.7
NS = (300, 1000, 3000, 10000)
KS = (1, 2, 5, 10, 20, 30)
REPS = 100
ETA = 0.05


def greedy_overfit(Z, kmax):
    """Tuning accuracy of each greedy prefix (ties as errors) for correctness Z (M, n)."""
    M_, n = Z.shape
    cnt = np.zeros(n, dtype=np.int16)
    remaining = list(range(M_))
    accs = []
    for k in range(1, kmax + 1):
        trial = cnt[None, :] + Z[remaining]                  # (|remaining|, n)
        acc = (2 * trial > k).mean(axis=1)
        b = int(np.argmax(acc))
        cnt += Z[remaining[b]]
        remaining.pop(b)
        accs.append(float(acc[b]))
    return np.asarray(accs)


def union_bound(k, n):
    return math.sqrt((k * math.log(math.e * M / k) + math.log(2 / ETA)) / (2 * n))


def lower_bound_k1(n):
    p = 1 - ACC
    sig = math.sqrt(n * p * (1 - p))
    x = math.sqrt(math.log(M / math.sqrt(8))) - 1 - 1 / sig
    ok = n * p >= sig * math.sqrt(math.log(M / math.sqrt(8))) + 1
    return math.sqrt(p * (1 - p) / n) * x if ok and x > 0 else None


def main(argv=None):
    rng = np.random.default_rng(0)
    kmax = max(KS)
    true = np.array([binom.sf(k // 2, k, ACC) for k in range(1, kmax + 1)])   # P[Bin > k/2]
    rows = []
    for n in NS:
        over = np.zeros((REPS, kmax))
        for r in range(REPS):
            Z = (rng.random((M, n)) < ACC).astype(np.int16)
            over[r] = greedy_overfit(Z, kmax) - true
        row = {"n": n, "lower_bound_k1": lower_bound_k1(n), "by_k": {}}
        for k in KS:
            v = over[:, k - 1]
            row["by_k"][str(k)] = {"mean": float(v.mean()), "q90": float(np.quantile(v, 0.9)),
                                   "bound": union_bound(k, n)}
        rows.append(row)
        print(f"n={n}: " + "; ".join(f"k={k} {row['by_k'][str(k)]['mean']:.4f} "
                                     f"(bound {row['by_k'][str(k)]['bound']:.4f})" for k in KS)
              + f"; k=1 lower bound {row['lower_bound_k1']}", flush=True)
    path = os.path.join(OUT, "pure_noise.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"M": M, "accuracy": ACC, "reps": REPS, "eta": ETA,
                   "true_accuracy": true.tolist(), "rows": rows}, f, indent=1)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
