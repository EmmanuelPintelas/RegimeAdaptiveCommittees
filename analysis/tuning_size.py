"""The decider and in-sample tuning greedy against the size of the tuning split.

For every independent pool with a tuning split, random subsamples of the tuning split of
each size (twenty draws below the full size, the full split once) are given to the
decider (Algorithm 1) and to greedy subset search at its best size on the subsample; both
are scored on the test split, probability averaging.  The proposer paths and the
collision estimate are the pool's, computed from the unlabeled test inputs.

    python -m analysis.tuning_size
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from . import selection
from .committees import AvgAgg, Scorer, greedy
from .pools import Pool, device
from .run import GPU_PROB_BUDGET, OUT, POOLS, Split, clean, log

SIZES = (300, 1000, 3000, 10000)
REPS = 20


def committee_score(agg, members, scorer, metric):
    agg.reset()
    for j in members:
        agg.add(j)
    return scorer(agg.pred, metric)


def analyse_pool(pool, r, seed=0):
    dev = device()
    paths = r["proposers"]["paths"]
    q = r["proposers"]["q_free"]
    met = pool.metric
    P_v = pool.probs(pool.tuning, where="cuda")
    sv = Split(pool, pool.tuning, dev)
    y_v, dom_v = sv.y, pool.domains(pool.tuning)
    on_gpu = pool.probs_bytes(pool.test) <= GPU_PROB_BUDGET
    P_t = pool.probs(pool.test, where="cuda" if on_gpu else "cpu")
    t_agg, t_scorer = AvgAgg(P_t, dev), Scorer(pool.labels(pool.test), pool.K, dev)
    N = len(y_v)
    rows_out = []
    for n in [s for s in SIZES if s < N] + [N]:
        for rep in range(REPS if n < N else 1):
            t0 = time.time()
            if n < N:
                rows = np.sort(np.random.default_rng((seed, n, rep)).choice(N, n, replace=False))
                r_t = torch.as_tensor(rows, device=dev)
                P_n, T_n = P_v[:, r_t], sv.T[:, r_t]
            else:
                rows, P_n, T_n = np.arange(N), P_v, sv.T
            y_n, dom_n = y_v[rows], dom_v[rows]
            v_agg, v_scorer = AvgAgg(P_n, dev), Scorer(y_n, pool.K, dev)

            def v_sub(rr, P_n=P_n):
                return AvgAgg(P_n[:, torch.as_tensor(rr, device=dev)], dev)

            d = selection.decide_main(pool, paths, v_agg, v_scorer, y_n, T_n, v_sub, dom_n, q)
            dec = committee_score(t_agg, d["members"], t_scorer, met)
            tg_order, tg_curve = greedy(v_agg, v_scorer, "balanced_acc")
            tg_k = int(np.argmax(tg_curve)) + 1
            tg = committee_score(t_agg, tg_order[:tg_k], t_scorer, met)
            rows_out.append({"n": int(n), "rep": rep, "source": d["source"], "k": d["k"],
                             "eps": d["eps"], "decider": dec, "tuning_greedy_k": tg_k,
                             "tuning_greedy": tg})
            log(f"  n={n} rep {rep}: decider {d['source']} k={d['k']} -> {dec:.4f}; "
                f"tuning-greedy k={tg_k} -> {tg:.4f}  [{time.time() - t0:.0f}s]")
            del v_agg, P_n, T_n
            torch.cuda.empty_cache()
    full = committee_score(t_agg, range(pool.M), t_scorer, met)
    del P_v, P_t, t_agg
    torch.cuda.empty_cache()
    return {"pool": pool.name, "benchmark": pool.benchmark, "metric": met, "n_tuning": N,
            "full_pool": full, "rows": rows_out}


def main(argv=None):
    path = os.path.join(OUT, "tuning_size.json")
    out = json.load(open(path, encoding="utf-8"))["pools"] if os.path.exists(path) else []
    done = {p["pool"] for p in out}
    for d in sorted(os.listdir(POOLS)):
        res_path = os.path.join(OUT, f"{d}.json")
        if not os.path.exists(res_path) or d in done:
            continue
        pool = Pool(os.path.join(POOLS, d))
        if not pool.tuning or pool.config.get("mode", "build") != "build":
            continue
        with open(res_path, encoding="utf-8") as f:
            r = json.load(f)
        log(f"=== {pool.name}")
        out.append(analyse_pool(pool, r))
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(clean({"pools": out}), f)
        os.replace(path + ".tmp", path)
    log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
