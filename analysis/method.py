"""Algorithm 1 with a tuning split: the decider and its terms (Algorithm 1).

Score.  Balanced accuracy on the tuning split V, over the classes present.  For a
candidate C of prior mass rho(C) its deviation is the weighted Hoeffding term of
Theorem 15,

    eps(C) = sqrt( log(2 / (eta rho(C))) * sum_c w_c^2 / (2 n_c) ),   w_c = 1 / |Y_V|,

valid conditionally on the class counts n_c for every class weighting fixed in
advance (accuracy is w_c = n_c / n).

Candidates and prior.  Half of the prior mass goes to the prefixes of the proposed
paths (uniform over paths), half to the search procedures held out across V's
domains (uniform over families, half per direction); within each, sizes follow

    rho_size(k | q) = 1/(2M) + (1/2) [ q 1{k <= k0} / k0 + (1 - q) 1{k > k0} / (M - k0) ],

k0 = ceil(sqrt(M)), q the label-free collision estimate on the unlabeled test inputs
(regime prior), or rho_size = 1/M (uniform prior).  The decider returns the candidate
of largest lower confidence bound.
"""
from __future__ import annotations

import math

import numpy as np

ETA = 0.05
FAMILIES = ("held-out greedy", "held-out top-k")


# --------------------------------------------------------------------------- #
# deviations and the prior
# --------------------------------------------------------------------------- #
def class_counts(y):
    c = np.bincount(np.asarray(y))
    return c[c > 0]


def balanced_variance(counts):
    """sum_c w_c^2 / (2 n_c) with w_c = 1/|Y|: the Hoeffding variance proxy of balanced accuracy."""
    counts = np.asarray(counts, dtype=float)
    return float(np.sum(1.0 / (2.0 * len(counts) ** 2 * counts)))


def eps_for_mass(counts, mass, eta=ETA):
    """Deviation of a candidate with prior mass ``mass``."""
    return math.sqrt(math.log(2.0 / (eta * mass)) * balanced_variance(counts))


def size_prior(M, q=None, k0=None):
    """rho_size(k), k = 1..M: uniform if q is None, else the regime prior."""
    if q is None:
        return np.full(M, 1.0 / M)
    k0 = int(math.ceil(math.sqrt(M))) if k0 is None else int(k0)
    q = float(min(max(q, 0.0), 1.0))
    k = np.arange(1, M + 1)
    small = (k <= k0) / k0
    large = (k > k0) / (M - k0)
    return 0.5 / M + 0.5 * (q * small + (1.0 - q) * large)


def spread_term(recalls):
    r = np.asarray(recalls, dtype=float)
    if r.min() <= 0:
        return float("inf")
    return float(((r - r.mean()) ** 2).mean() / (2 * r.min()))


def per_proposer(v_curves, t_curves, metric):
    """Each proposer at the size best on V (balanced accuracy), scored on the test split."""
    out = {}
    for name in v_curves:
        vb = v_curves[name]["balanced_acc"]
        k = int(np.argmax(vb)) + 1
        out[name] = {"k_V": k, "V_balanced_acc": vb[k - 1],
                     "test_at_kV": t_curves[name][metric][k - 1],
                     "test_best": max(t_curves[name][metric]),
                     "test_best_k": int(np.argmax(t_curves[name][metric])) + 1}
    return out


# --------------------------------------------------------------------------- #
# held-out halves
# --------------------------------------------------------------------------- #
def domain_halves(domains, seed=0):
    """Two halves of V for held-out subset search.

    Whole domains, assigned largest first to the lighter half, when V spans several;
    random halves of the points when it spans one.  Returns (rows A, rows B, note).
    """
    d = np.asarray(domains)
    u, c = np.unique(d, return_counts=True)
    if len(u) >= 2:
        groups, load = ([], []), [0, 0]
        for dom, n in sorted(zip(u.tolist(), c.tolist()), key=lambda t: (-t[1], t[0])):
            g = int(np.argmin(load))
            groups[g].append(dom)
            load[g] += n
        a = np.isin(d, groups[0])
        note = f"{len(u)} domains, {len(groups[0])}+{len(groups[1])}"
    else:
        a = np.random.default_rng(seed).random(len(d)) < 0.5
        note = "one domain, random halves"
    return np.flatnonzero(a), np.flatnonzero(~a), note


# --------------------------------------------------------------------------- #
# the decider
# --------------------------------------------------------------------------- #
def decide(prop_scores, held, counts_v, M, q=None, eta=ETA, kmax=None, fixed=None):
    """The decider: largest lower confidence bound over every candidate.

    prop_scores  {path: [balanced accuracy on V of prefix k, k = 1..M]}, fixed before V.
    held         {family: {"score": [held-out score at k], "halves": [(w_h, counts of the
                 half that scores it), ...]}}, each half's path fixed before those labels.
    fixed        optional {name: balanced accuracy on V} of single fixed committees
                 (e.g. the full pool alone), which then take the proposed half of the
                 prior mass uniformly in place of the paths.
    q            the label-free collision estimate (regime prior) or None (uniform).
    kmax         a deployment budget: only sizes <= kmax are candidates.

    Returns {"source", "k", "score", "lcb", "eps"}; source is a path, a fixed
    committee, or a search family (refit on all of V at size k).
    """
    rho_k = size_prior(M, q)
    kmax = M if kmax is None else int(kmax)
    best = None

    def consider(source, k, s, e):
        nonlocal best
        if best is None or s - e > best["lcb"] + 1e-12:
            best = {"source": source, "k": int(k), "score": float(s), "lcb": float(s - e),
                    "eps": float(e)}

    groups = [g for g in (fixed or prop_scores, held) if g]
    group_mass = 1.0 / len(groups)
    if fixed:
        for name, s in fixed.items():
            consider(name, M, s, eps_for_mass(counts_v, group_mass / len(fixed), eta))
    else:
        for name, curve in (prop_scores or {}).items():
            for k in range(1, min(kmax, len(curve)) + 1):
                mass = group_mass / len(prop_scores) * rho_k[k - 1]
                consider(name, k, curve[k - 1], eps_for_mass(counts_v, mass, eta))
    for fam, h in (held or {}).items():
        for k in range(1, min(kmax, len(h["score"])) + 1):
            mass = group_mass / len(held) * rho_k[k - 1] / len(h["halves"])
            e = sum(w * eps_for_mass(c, mass, eta) for w, c in h["halves"])
            consider(fam, k, h["score"][k - 1], e)
    return best
