"""Sensitivity of the decider to its score: balanced accuracy (the method) against accuracy.

Algorithm 1 scores candidates by balanced accuracy on the tuning split.  This module reruns
the same decider (same menu, both held-out search families, regime prior, deviations by
the weighted Hoeffding term with the matching class weights) with accuracy as the score,
w_c = n_c / n, and scores the choice on the test split.  It is a sensitivity analysis of
the frozen method, not a change to it.  Probability averaging.

    python -m analysis.score_ablation
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

from . import method, proposers
from .committees import AvgAgg, Scorer, greedy, member_scores, path_scores
from .pools import Pool, device
from .run import GPU_PROB_BUDGET, OUT, POOLS, Split, clean, log


def eps_acc(n, mass, eta=method.ETA):
    """Weighted Hoeffding deviation with w_c = n_c / n: sqrt(log(2/(eta mass)) / (2n))."""
    return math.sqrt(math.log(2.0 / (eta * mass)) / (2.0 * n))


def decide_acc(prop, held, n_v, M, q, eta=method.ETA):
    rho_k = method.size_prior(M, q)
    best = None

    def consider(source, k, s, e):
        nonlocal best
        if best is None or s - e > best["lcb"] + 1e-12:
            best = {"source": source, "k": int(k), "score": float(s), "lcb": float(s - e),
                    "eps": float(e)}
    for name, curve in prop.items():
        for k in range(1, len(curve) + 1):
            consider(name, k, curve[k - 1], eps_acc(n_v, 0.5 / len(prop) * rho_k[k - 1], eta))
    for fam, h in held.items():
        for k in range(1, len(h["score"]) + 1):
            mass = 0.5 / len(held) * rho_k[k - 1] / len(h["halves"])
            consider(fam, k, h["score"][k - 1], sum(w * eps_acc(n_h, mass, eta)
                                                    for w, n_h in h["halves"]))
    return best


def analyse_pool(pool, r):
    dev = device()
    met = pool.metric
    paths = r["proposers"]["paths"]
    q = r["proposers"]["q_free"]
    sv = Split(pool, pool.tuning, dev)
    y_v, dom_v = sv.y, pool.domains(pool.tuning)
    P_v = pool.probs(pool.tuning, where="cuda")
    v_agg, v_sc = AvgAgg(P_v, dev), Scorer(y_v, pool.K, dev)
    A, B, _ = method.domain_halves(dom_v)
    held = {f: {"score": np.zeros(pool.M), "halves": []} for f in method.FAMILIES}
    for fit, sc in ((A, B), (B, A)):
        f_t = torch.as_tensor(fit, device=dev)
        fit_sc = Scorer(y_v[fit], pool.K, dev)
        orders = {"held-out greedy": greedy(AvgAgg(P_v[:, f_t], dev), fit_sc, "acc")[0],
                  "held-out top-k": [int(j) for j in np.argsort(
                      -member_scores(sv.T[:, f_t], fit_sc, "acc"), kind="stable")]}
        agg = AvgAgg(P_v[:, torch.as_tensor(sc, device=dev)], dev)
        sc_sc = Scorer(y_v[sc], pool.K, dev)
        w = len(sc) / len(y_v)
        for f, o in orders.items():
            held[f]["score"] += w * np.asarray(path_scores(agg, o, sc_sc, ("acc",))["acc"])
            held[f]["halves"].append((w, len(sc)))
        del agg
        torch.cuda.empty_cache()
    prop = {n: path_scores(v_agg, o, v_sc, ("acc",))["acc"]
            for n, o in proposers.menu(paths).items()}
    dec = decide_acc(prop, held, len(y_v), pool.M, q)
    if dec["source"] == "held-out greedy":
        members = greedy(v_agg, v_sc, "acc")[0][:dec["k"]]
    elif dec["source"] == "held-out top-k":
        members = [int(j) for j in np.argsort(-member_scores(sv.T, v_sc, "acc"),
                                              kind="stable")[:dec["k"]]]
    else:
        members = paths[dec["source"]][:dec["k"]]
    on_gpu = pool.probs_bytes(pool.test) <= GPU_PROB_BUDGET
    P_t = pool.probs(pool.test, where="cuda" if on_gpu else "cpu")
    t_agg, t_sc = AvgAgg(P_t, dev), Scorer(pool.labels(pool.test), pool.K, dev)
    test = {m: v[-1] for m, v in path_scores(t_agg, members, t_sc, (met, "acc")).items()}
    full = r["method"]["avg"]["full_pool_test"]
    return {"pool": pool.name, "benchmark": pool.benchmark, "metric": met,
            "accuracy_score": {**dec, "members": members, "test": test},
            "balanced_score": {k: r["method"]["avg"]["main"][k] for k in ("source", "k", "eps", "test")},
            "full_pool": full}


def main(argv=None):
    out = []
    for d in sorted(os.listdir(POOLS)):
        path = os.path.join(OUT, f"{d}.json")
        if not os.path.exists(path):
            continue
        pool = Pool(os.path.join(POOLS, d))
        if not pool.tuning or pool.config.get("mode", "build") != "build":
            continue
        r = json.load(open(path, encoding="utf-8"))
        t0 = time.time()
        res = analyse_pool(pool, r)
        out.append(res)
        a, b = res["accuracy_score"], res["balanced_score"]
        log(f"{pool.name}: accuracy {a['source']} k={a['k']} {a['test'][pool.metric]:.4f}; balanced "
            f"{b['source']} k={b['k']} {b['test'][pool.metric]:.4f}; full "
            f"{res['full_pool'][pool.metric]:.4f}  [{time.time() - t0:.0f}s]")
    with open(os.path.join(OUT, "score_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(clean({"pools": out}), f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
