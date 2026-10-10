"""Shared ancestry (Proposition 12): refined pools against independent ones.

A refined pool holds the children of the first twenty members of a benchmark's
EfficientNet-B0 pool, five each, fine-tuned for one or two epochs.  On the test split,
each parent splits the points into its mistakes and the rest.  On both parts the children
of one parent (siblings) are compared with independent members of the parent's own pool
(its members other than the twenty parents): their error rate, and the collision
probability of their wrong votes, the share of pairs of members wrong at a point that
agree on the wrong label, pooled over pairs, points and parents.  The proposition's copy
rate theta is estimated from the error rates on the parent's mistakes, where a child is
right with probability (1 - theta) p, p estimated by the independent members' accuracy
there.  Each pool's label-optimal size is that of greedy selection with the test labels,
plurality vote, the pool metric.  Writes results/main/ancestry.json.

    python -m analysis.ancestry
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from . import geometry, method
from .committees import Scorer, VoteAgg, greedy, path_scores
from .pools import Pool, device
from .run import OUT, POOLS, clean, log

BENCHMARKS = ("camelyon17", "kather", "iwildcam", "rxrx1")


def counts(T, y, K):
    """(same-label wrong pairs, wrong pairs, wrong votes, votes) of members T on points y."""
    if T.shape[1] == 0:
        return np.zeros(4)
    c = geometry.disagreement_counts(T, y, K).double()
    w = c.sum(dim=1)
    return np.array([float((c * (c - 1) / 2).sum()), float((w * (w - 1) / 2).sum()),
                     float((T != y[None]).sum()), float(T.numel())])


def best_size(T, C, y, K, metric, dev, halves=None):
    """Greedy reference size, in-sample and cross-fitted over the given halves of the points."""
    y_np = y.cpu().numpy()
    sc = Scorer(y_np, K, dev)
    order, curve = greedy(VoteAgg(T, C, K), sc, metric)
    k = int(np.argmax(curve)) + 1
    out = {"k": k, "best": float(curve[k - 1]), "full": float(curve[-1]), "members": order[:k]}
    if halves is not None:
        cf = []
        for fit, ev in halves:
            f, e = (torch.as_tensor(r, device=dev) for r in (fit, ev))
            o, c = greedy(VoteAgg(T[:, f], C[:, f], K), Scorer(y_np[fit], K, dev), metric)
            kf = int(np.argmax(c)) + 1
            ce = path_scores(VoteAgg(T[:, e], C[:, e], K), o, Scorer(y_np[ev], K, dev),
                             (metric,))[metric]
            cf.append({"k": kf, "members": o[:kf], "best": float(ce[kf - 1]),
                       "full": float(ce[-1])})
        out["crossfit"] = cf
    return out


def analyse(bench, dev):
    R = Pool(os.path.join(POOLS, f"{bench}_refined_seed_101"))
    I = Pool(os.path.join(POOLS, f"{bench}_seed_1"))
    split = R.test
    y = torch.as_tensor(R.labels(split), device=dev)
    cr, ci = R.compact(split, log=log), I.compact(split, log=log)
    TR = torch.as_tensor(cr["top1"].astype(np.int64), device=dev)
    TI = torch.as_tensor(ci["top1"].astype(np.int64), device=dev)
    parents = [row["parent"]["member_id"] for row in R.rows]
    parent_ids = sorted(set(parents))
    pos_I = {m: i for i, m in enumerate(I.ids)}
    indep = torch.as_tensor([i for i, m in enumerate(I.ids) if m not in set(parent_ids)],
                            device=dev)
    acc = {(g, part): np.zeros(4) for g in ("siblings", "independent") for part in ("H", "E")}
    for p in parent_ids:
        wrong = TI[pos_I[p]] != y
        kids = torch.as_tensor([i for i, q in enumerate(parents) if q == p], device=dev)
        for part, mask in (("H", wrong), ("E", ~wrong)):
            acc[("siblings", part)] += counts(TR[kids][:, mask], y[mask], R.K)
            acc[("independent", part)] += counts(TI[indep][:, mask], y[mask], R.K)

    def stats(g, part):
        same, both, wv, n = acc[(g, part)]
        return {"collision": same / both if both else None, "error": wv / n if n else None}
    out = {g: {part: stats(g, part) for part in ("H", "E")} for g in ("siblings", "independent")}
    p_ind = 1 - out["independent"]["H"]["error"]
    out["theta"] = 1 - (1 - out["siblings"]["H"]["error"]) / p_ind if p_ind > 0 else None
    out["parent_error"] = float(np.mean([float((TI[pos_I[p]] != y).float().mean())
                                         for p in parent_ids]))
    CR = torch.as_tensor(cr["conf"].astype(np.float32), device=dev)
    CI = torch.as_tensor(ci["conf"].astype(np.float32), device=dev)
    A, B, _ = method.domain_halves(R.domains(split), seed=0)
    halves = ((A, B), (B, A))
    out["size_refined"] = best_size(TR, CR, y, R.K, R.metric, dev, halves)
    out["size_refined"]["distinct_parents"] = len({parents[i] for i in out["size_refined"]["members"]})
    for c in out["size_refined"]["crossfit"]:
        c["distinct_parents"] = len({parents[i] for i in c["members"]})
    out["size_independent"] = best_size(TI, CI, y, I.K, I.metric, dev, halves)
    out.update(benchmark=bench, K=R.K, metric=R.metric, split=split, parents=len(parent_ids),
               refined_pool=R.name, independent_pool=I.name, n_independent=int(len(indep)))
    return out


def main(argv=None):
    dev = device()
    res = []
    for b in BENCHMARKS:
        if not os.path.exists(os.path.join(POOLS, f"{b}_refined_seed_101", "predictions",
                                           "index.json")):
            continue
        t0 = time.time()
        r = analyse(b, dev)
        res.append(r)
        s, i = r["siblings"], r["independent"]
        log(f"{b}: error on the parent's mistakes siblings {s['H']['error']:.3f} vs independent "
            f"{i['H']['error']:.3f}, elsewhere {s['E']['error']:.3f} vs {i['E']['error']:.3f}; "
            f"collision on mistakes {s['H']['collision']:.3f} vs {i['H']['collision']:.3f}, "
            f"elsewhere {s['E']['collision']} vs {i['E']['collision']}; theta {r['theta']:.3f}; "
            f"best k refined {r['size_refined']['k']} vs independent {r['size_independent']['k']}"
            f"  [{time.time() - t0:.0f}s]")
    path = os.path.join(OUT, "ancestry.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean({"benchmarks": res}), f)
    log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
