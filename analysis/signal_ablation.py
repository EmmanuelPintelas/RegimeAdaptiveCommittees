"""Ablation of the signals in the method's consensus (P11, P12), under Algorithm 1.

For every independent pool with a tuning split and every subset of the five per-model
signals (confidence P1, spectral exponent P2, agreement with the pool P3, source
validation P6, weight-noise stability P8), the consensus paths are rebuilt from that
subset alone, a mean and a geometric consensus per spectral direction, and the decider
is run twice: as the method, on the menu {P11, P12, P7} with both held-out search
families and the regime prior; and as the consensus alone, on {P11, P12} without P7 and
without search.  The empty subset leaves P7 and search, or, for the consensus alone, the
full pool.  A mixed pool has no spectral exponent, so there a subset with it is the same
subset without it.  Probability averaging; test score minus the full pool's.

    python -m analysis.signal_ablation
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np

from . import method, proposers, selection
from .committees import AvgAgg, Scorer, greedy, member_scores, path_scores
from .pools import Pool, device
from .run import GPU_PROB_BUDGET, OUT, POOLS, Split, clean, log

SIGNALS = ("P1", "P2", "P3", "P6", "P8")


def analyse_pool(pool, r):
    dev = device()
    met = pool.metric
    m = r["method"]["avg"]
    full = m["full_pool_test"][met]
    paths = r["proposers"]["paths"]
    q = r["proposers"]["q_free"]
    sv = Split(pool, pool.tuning, dev)
    y_v, dom_v = sv.y, pool.domains(pool.tuning)
    counts = method.class_counts(y_v)
    import torch
    P_v = pool.probs(pool.tuning, where="cuda")
    v_agg, v_sc = AvgAgg(P_v, dev), Scorer(y_v, pool.K, dev)

    def v_sub(rr):
        return AvgAgg(P_v[:, torch.as_tensor(rr, device=dev)], dev)

    held, _ = selection.held_out_families(pool.M, pool.K, sv.T, v_sub, y_v, dom_v, dev)
    refit = {"held-out greedy": greedy(v_agg, v_sc, "balanced_acc")[0],
             "held-out top-k": [int(j) for j in np.argsort(
                 -member_scores(sv.T, v_sc, "balanced_acc"), kind="stable")]}
    on_gpu = pool.probs_bytes(pool.test) <= GPU_PROB_BUDGET
    P_t = pool.probs(pool.test, where="cuda" if on_gpu else "cpu")
    t_agg, t_sc = AvgAgg(P_t, dev), Scorer(pool.labels(pool.test), pool.K, dev)
    v_cache, t_cache = {}, {}

    def v_curve(order):
        key = tuple(order)
        if key not in v_cache:
            v_cache[key] = path_scores(v_agg, order, v_sc, ("balanced_acc",))["balanced_acc"]
        return v_cache[key]

    def t_score(members):
        key = tuple(members)
        if key not in t_cache:
            t_agg.reset()
            for j in members:
                t_agg.add(j)
            t_cache[key] = t_sc(t_agg.pred, met)
        return t_cache[key]

    def members_of(dec, menu):
        if dec["source"] in method.FAMILIES:
            return refit[dec["source"]][:dec["k"]]
        return menu[dec["source"]][:dec["k"]]

    out = {}
    p7 = {"P7 collision-aware": paths["P7 collision-aware"]}
    for size in range(len(SIGNALS) + 1):
        for subset in itertools.combinations(SIGNALS, size):
            c = proposers.consensus(paths, pool.M, subset)
            cons = {f"P11{s}": om for s, (om, _) in c.items()}
            cons.update({f"P12{s}": og for s, (_, og) in c.items()})
            row = {"paths": len(cons)}
            menu = {**p7, **cons}
            dec = method.decide({n: v_curve(o) for n, o in menu.items()}, held, counts, pool.M,
                                q=q)
            row.update({"with": t_score(members_of(dec, menu)) - full,
                        "with_source": dec["source"], "with_k": dec["k"]})
            if cons:
                dec = method.decide({n: v_curve(o) for n, o in cons.items()}, None, counts,
                                    pool.M, q=q)
                row.update({"without": t_score(members_of(dec, cons)) - full,
                            "without_source": dec["source"], "without_k": dec["k"]})
            else:
                row.update({"without": 0.0, "without_source": "full pool", "without_k": pool.M})
            out["+".join(subset) or "none"] = row
    allfive = out["+".join(SIGNALS)]
    check = abs(allfive["with"] - (m["main"]["test"][met] - full))
    return {"pool": pool.name, "benchmark": pool.benchmark, "metric": met, "full_pool": full,
            "subsets": out, "check_against_method": check}


def main(argv=None):
    res = []
    for d in sorted(os.listdir(POOLS)):
        path = os.path.join(OUT, f"{d}.json")
        if not os.path.exists(path):
            continue
        pool = Pool(os.path.join(POOLS, d))
        if not pool.tuning or pool.config.get("mode", "build") != "build":
            continue
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        t0 = time.time()
        out = analyse_pool(pool, r)
        res.append(out)
        log(f"{pool.name}: 32 subsets, all five {out['subsets']['+'.join(SIGNALS)]['with']:+.4f} "
            f"(check against the method {out['check_against_method']:.1e})  "
            f"[{time.time() - t0:.0f}s]")
    path = os.path.join(OUT, "signal_ablation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean({"signals": SIGNALS, "pools": res}), f)
    log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
