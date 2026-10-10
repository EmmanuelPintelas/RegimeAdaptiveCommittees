"""Random-committee error against size: measured, and predicted by the theory.

Measured.  Committees of size k drawn uniformly without replacement from the
pool (the prefixes of random permutations), plurality vote, averaged over
permutations.  The tie rule is a uniform random tie-break, whose expected error is
computed exactly from the vote counts (``tie="random"``), or ties counted as errors
(``tie="error"``).

Predicted.  At each point x only two things are used: how many of the M members
are right there, M_c(x), and one number for the whole split, the pooled
collision probability q.  A committee of size k then has N ~ Hypergeometric(M,
M_c(x), k) correct votes, its k - N wrong votes fall on L = round(1/q) equally
likely wrong labels, and it errs with the probability the tie rule gives for
(N, largest wrong count, its multiplicity).  For two classes L = 1 and this is
exact, which the pipeline checks.  The same prediction with L = K - 1 is the one
the class count alone would make.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import hypergeom

from .committees import PluralityCounter


def measured_random(top1, y, K, R=30, seed=0, tie="random"):
    """Zero-one and balanced error of random committees, k = 1..M, mean and sd over R."""
    M = top1.shape[0]
    rng = np.random.default_rng(seed)
    counter = PluralityCounter(top1, y, K)
    err = np.zeros((R, M))
    berr = np.zeros((R, M))
    for r in range(R):
        counter.reset()
        for k, j in enumerate(rng.permutation(M)):
            counter.add(int(j))
            err[r, k], berr[r, k] = counter.errors(tie)
    return {"err_mean": err.mean(0).tolist(), "err_sd": err.std(0).tolist(),
            "berr_mean": berr.mean(0).tolist(), "tie": tie}


def _max_and_multiplicity(w, L, sims, rng):
    """Largest count m of w votes over L equally likely labels, and how many labels reach it."""
    if L == 1:
        return np.full(sims, w), np.ones(sims, dtype=np.int64)
    labels = rng.integers(0, L, size=(sims, w))
    counts = np.zeros((sims, L), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(sims), w), labels.ravel()), 1)
    m = counts.max(axis=1)
    return m, (counts == m[:, None]).sum(axis=1)


def error_table(M, L, sims=4000, seed=0, tie="random"):
    """F[w, n]: error of a plurality with n correct and w wrong votes on L uniform labels.

    tie="error":  P(m >= n), ties counted as errors.
    tie="random": P(m > n) + E[r / (r + 1); m = n], a uniform random tie-break among
                  the n correct votes' label and the r wrong labels with m = n votes.
    F[w, 0] = 1 for w >= 1 and F[0, n] = 0 for n >= 1; the empty committee counts as
    an error.
    """
    rng = np.random.default_rng(seed)
    F = np.zeros((M + 1, M + 2))
    F[0, 0] = 1.0
    for w in range(1, M + 1):
        m, r = _max_and_multiplicity(w, L, sims, rng)
        n = np.arange(M + 2)
        above = (m[None, :] > n[:, None]).mean(axis=1)
        if tie == "error":
            F[w] = above + (m[None, :] == n[:, None]).mean(axis=1)
        else:
            share = r / (r + 1.0)
            F[w] = above + ((m[None, :] == n[:, None]) * share[None, :]).mean(axis=1)
        F[w, 0] = 1.0
    return F


def predicted_random(correct_counts, y, M, L, ks=None, sims=4000, seed=0, tie="random"):
    """Predicted zero-one and balanced error of random committees under (M_c(x), L)."""
    ks = list(range(1, M + 1)) if ks is None else list(ks)
    F = error_table(M, max(1, int(L)), sims=sims, seed=seed, tie=tie)
    mc = np.asarray(correct_counts, dtype=np.int64)
    groups, inverse = np.unique(mc, return_inverse=True)
    y = np.asarray(y)
    classes, cls_idx = np.unique(y, return_inverse=True)
    err, berr = [], []
    for k in ks:
        n = np.arange(k + 1)
        pmf = hypergeom.pmf(n[None, :], M, groups[:, None], k)            # (groups, k+1)
        given = F[k - n, n]
        per_group = (pmf * given[None, :]).sum(axis=1)
        per_point = per_group[inverse]
        err.append(float(per_point.mean()))
        acc_c = np.bincount(cls_idx, weights=1.0 - per_point) / np.bincount(cls_idx)
        berr.append(float(1.0 - acc_c.mean()))
    return {"ks": ks, "L": int(L), "err": err, "berr": berr, "tie": tie}


def correct_counts(top1, y):
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=top1.device)
    return (top1 == yt[None, :]).sum(dim=0).cpu().numpy()
