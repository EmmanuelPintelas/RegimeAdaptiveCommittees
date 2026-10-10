"""The label-free proposer menu of Section 5.

Every proposer reads only label-free information: the members' top-1 labels and
confidences on the unlabeled test inputs, their source-validation scores, and
their weight statistics.  Each returns an ordering of the pool, whose prefixes
are the candidate committees.

    P1  confidence                  mean top-1 probability, descending
    P2  spectral exponent           alpha, ascending and descending (single-backbone pools)
    P3  pool agreement              agreement with the full-pool plurality z
    P4  pseudo-label complementarity  greedy, committee agreement with z
    P5  confidence + lambda * disagreement with the current committee, per lambda
    P6  source validation           source-validation score, descending
    P7  collision-aware greedy      greedy on the committee error under the
                                    collision model (Theorem 10): each member is
                                    right with probability equal to its own
                                    confidence at the point, wrong votes fall on
                                    L = round(1/q) equally likely labels with q the
                                    label-free collision estimate, and ties count
                                    half; its own size is the prefix of smallest
                                    modelled error
    P8  weight-noise stability      share of unlabeled target images whose predicted
                                    class flips under per-tensor scaled Gaussian
                                    weight noise, ascending; one path per noise
                                    scale (analysis.weight_noise)
    P9  consensus, mean             rank consensus of the proposer families P1-P8:
    P10 consensus, geometric        each path's ordering becomes a rank quality
                                    q = 1 - (rank - 1)/(M - 1), the paths of one
                                    family (P5's lambdas, P8's scales) are averaged
                                    into one family score, and the family scores
                                    are combined by their mean (P9) or geometric
                                    mean (P10, with q + 1/(M - 1))
    P11 signals, mean               the same over the per-model signals only (P1,
    P12 signals, geometric          P2, P3, P6, P8); the menu of the method

The two spectral directions are never averaged, since they would cancel: on a
single-backbone pool every consensus comes in two versions, one per direction.
"""
from __future__ import annotations

import numpy as np
import torch

from . import geometry, predict, weight_noise
from .committees import VoteAgg

LAMBDAS = (0.25, 0.5, 1.0, 2.0)
ALL_FAMILIES = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8")
SIGNAL_FAMILIES = ("P1", "P2", "P3", "P6", "P8")
# menus compared by the ablation, as the proposer numbers each admits.  The menu of the
# method is the consensus of the per-model signals, P11-P12, plus the theory-derived
# path P7, whose size the decider sets: few paths, and no proposer built by a greedy
# search without theory behind it.  The others are ablations; the consensus menus are
# also decided without subset search (NO_SUBSET_MENUS)
MENUS = {"P1-P7": tuple(range(1, 8)), "P1-P8": tuple(range(1, 9)),
         "P1-P7,P9-P10": (1, 2, 3, 4, 5, 6, 7, 9, 10),
         "P1-P10": tuple(range(1, 11)), "P1-P12": tuple(range(1, 13)),
         "P9-P12": (9, 10, 11, 12), "P11-P12": (11, 12), "P7,P11-P12": (7, 11, 12)}
NO_SUBSET_MENUS = ("P9-P12", "P11-P12", "P7,P11-P12")
MAIN_MENU = "P7,P11-P12"


def proposer_number(name):
    return int(name.split()[0][1:])


def menu(paths, name=MAIN_MENU):
    """The paths of one menu, in menu order."""
    keep = MENUS[name]
    return {n: o for n, o in paths.items() if proposer_number(n) in keep}


def consensus(paths, M, families):
    """{direction suffix: (mean order, geometric order)} of the rank consensus of ``families``.

    Each path's ordering becomes a rank quality q = 1 - (rank - 1)/(M - 1); the paths of
    one family (P5's lambdas, P8's scales) are averaged into one family score; P2 enters
    once per direction, so with P2 among the families on a single-backbone pool there are
    two versions.  Families the pool does not have are skipped; with none left the result
    is empty."""
    pos = {}
    for name, order in paths.items():
        if name.split()[0] in families:
            q = np.empty(M)
            q[np.asarray(order)] = 1.0 - np.arange(M) / (M - 1)
            pos[name] = q
    family = {}
    for name, q in pos.items():
        if name.split()[0] != "P2":
            family.setdefault(name.split()[0], []).append(q)
    family = {f: np.mean(v, axis=0) for f, v in family.items()}
    directions = {"": None}
    if "P2 alpha low" in pos:
        directions = {" alpha low": pos["P2 alpha low"], " alpha high": pos["P2 alpha high"]}
    eps = 1.0 / (M - 1)
    out = {}
    for suffix, p2 in directions.items():
        qs = [family[f] for f in families if f in family] + ([p2] if p2 is not None else [])
        if not qs:
            continue
        qs = np.asarray(qs)
        out[suffix] = (list(np.argsort(-qs.mean(0), kind="stable")),
                       list(np.argsort(-np.exp(np.log(qs + eps).mean(0)), kind="stable")))
    return out


def consensus_paths(paths, M):
    """P9-P12: rank consensus over proposer families, mean and geometric."""
    out = {}
    for tag_mean, tag_geo, fams, label in (("P9", "P10", ALL_FAMILIES, "consensus"),
                                           ("P11", "P12", SIGNAL_FAMILIES, "signals")):
        c = consensus(paths, M, fams)
        for suffix, (o_mean, _) in c.items():
            out[f"{tag_mean} {label} mean{suffix}"] = o_mean
        for suffix, (_, o_geo) in c.items():
            out[f"{tag_geo} {label} geo{suffix}"] = o_geo
    return out


def pool_plurality(T, C, K):
    agg = VoteAgg(T, C, K)
    for j in range(T.shape[0]):
        agg.add(j)
    return agg.pred


def collision_greedy(T, ref, q, start):
    """Greedy on the marginal value (8) with 0/1 plug-ins against ``ref``.

    Returns the full ordering, the stopping size (the committee size before the
    first addition whose best marginal value is non-negative), and the best
    marginal value at each step.
    """
    M = T.shape[0]
    ref = torch.as_tensor(ref, device=T.device) if not torch.is_tensor(ref) else ref
    phat = (T == ref[None, :]).float()
    wrong = 1.0 - phat
    order = [int(start)]
    n_c = phat[start].clone()
    w_c = wrong[start].clone()
    remaining = [j for j in range(M) if j != start]
    deltas, stop = [], None
    while remaining:
        idx = torch.as_tensor(remaining, device=T.device)
        knife = (n_c == 1).float()
        benefit = (phat[idx] * knife[None, :]).mean(dim=1)
        cost = q * (wrong[idx] * w_c[None, :]).mean(dim=1)
        delta = cost - benefit
        b = int(torch.argmin(delta))
        j, d = remaining[b], float(delta[b])
        if stop is None and d >= 0:
            stop = len(order)
        order.append(j)
        deltas.append(d)
        n_c += phat[j]
        w_c += wrong[j]
        remaining.pop(b)
    return order, (stop if stop is not None else M), deltas


def model_table(M, L, dev, sims=20000):
    """E[n, w]: plurality error with n correct and w wrong votes, uniform random tie-break.

    Wrong votes fall independently on L equally likely labels; a tie between the
    correct label and r wrong labels is lost with probability r / (r + 1).  Exact for
    two classes (L = 1).  The empty committee (n = w = 0) counts as an error.
    """
    F = predict.error_table(M, max(1, int(L)), sims=sims, seed=0, tie="random")  # F[w, n]
    E = F[:, :M + 1].T.copy()                                                    # E[n, w]
    return torch.as_tensor(E, dtype=torch.float32, device=dev)


def model_greedy(B, E, chunk=10):
    """Greedy on the modelled committee error.

    ``B`` is (M, S, N) uint8: S draws of each member's correctness at each point
    (S = 1 for known correctness).  Returns the ordering, the modelled error of
    every prefix, and the prefix size of smallest modelled error.
    """
    M, S, N = B.shape
    dev = B.device
    Ef = E.reshape(-1)
    n_c = torch.zeros((S, N), dtype=torch.long, device=dev)
    w_c = torch.zeros((S, N), dtype=torch.long, device=dev)
    remaining, order, errors = list(range(M)), [], []
    while remaining:
        best_e, best_j = None, None
        for c0 in range(0, len(remaining), chunk):
            idx = torch.as_tensor(remaining[c0:c0 + chunk], device=dev)
            b = B[idx].long()
            e = Ef[(n_c[None] + b) * (M + 1) + (w_c[None] + 1 - b)].mean(dim=(1, 2))
            k = int(torch.argmin(e))
            if best_e is None or float(e[k]) < best_e:
                best_e, best_j = float(e[k]), remaining[c0 + k]
        b = B[best_j].long()
        n_c += b
        w_c += 1 - b
        order.append(best_j)
        errors.append(best_e)
        remaining.remove(best_j)
    return order, errors, int(np.argmin(errors)) + 1


def confidence_draws(C, S=16, seed=0):
    g = torch.Generator(device=C.device).manual_seed(seed)
    return (torch.rand((C.shape[0], S, C.shape[1]), generator=g, device=C.device)
            < C[:, None, :]).to(torch.uint8)


def _greedy_label_free(T, C, K, gain):
    """Forward selection maximising ``gain(agg, j)``, a label-free criterion."""
    M = T.shape[0]
    agg = VoteAgg(T, C, K)
    remaining, order = list(range(M)), []
    while remaining:
        scores = [gain(agg, j) for j in remaining]
        j = remaining[int(np.argmax(scores))]
        agg.add(j)
        remaining.remove(j)
        order.append(j)
    return order


def proposer_paths(pool, T, C, noise=None):
    """All proposer orderings from the test split's unlabeled inputs, plus P7 details.

    ``noise`` overrides the cached weight-noise flip rates (P8), for the inductive check."""
    K = pool.K
    z = pool_plurality(T, C, K)
    conf = C.float().mean(dim=1).cpu().numpy()
    agree = (T == z[None, :]).float().mean(dim=1).cpu().numpy()
    paths = {"P1 confidence": list(np.argsort(-conf, kind="stable"))}
    if pool.single_arch():
        alpha = pool.weight_stat("alpha")
        if np.isfinite(alpha).all():
            paths["P2 alpha low"] = list(np.argsort(alpha, kind="stable"))
            paths["P2 alpha high"] = list(np.argsort(-alpha, kind="stable"))
    paths["P3 agreement"] = list(np.argsort(-agree, kind="stable"))
    paths["P4 complementarity"] = _greedy_label_free(
        T, C, K, lambda agg, j: float((agg.trial(j) == z).float().mean()))
    conf_t = torch.as_tensor(conf, device=T.device)
    for lam in LAMBDAS:
        def gain(agg, j, lam=lam):
            if not agg.members:
                return float(conf_t[j])
            return float(conf_t[j] + lam * (T[j] != agg.pred).float().mean())
        paths[f"P5 conf+{lam:g}div"] = _greedy_label_free(T, C, K, gain)
    paths["P6 source val"] = list(np.argsort(-pool.source_val_score(), kind="stable"))
    q_free = geometry.collision_stats(T, z, K)["q"]
    L_free = max(1, round(1 / q_free)) if q_free and q_free > 0 else 1
    E = model_table(pool.M, L_free, T.device)
    B = confidence_draws(C)
    order, errors, stop = model_greedy(B, E)
    del B
    paths["P7 collision-aware"] = order
    noise = weight_noise.load(pool) if noise is None else noise
    if noise is None:
        raise SystemExit(f"{pool.name}: no weight-noise cache; run python -m analysis.weight_noise")
    for sigma, flips in sorted(noise.items()):
        paths[f"P8 noise {sigma:g}"] = list(np.argsort(flips, kind="stable"))
    paths.update(consensus_paths(paths, pool.M))
    paths = {k: [int(v) for v in o] for k, o in paths.items()}
    return paths, {"z": z, "q_free": q_free, "L_free": L_free, "p7_stop": int(stop),
                   "p7_model_errors": errors}
