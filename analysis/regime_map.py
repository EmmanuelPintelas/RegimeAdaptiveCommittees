"""The regime map: collision, concentration, label-optimal size and gain, per cell.

A cell is one pool's test split restricted to a set of classes S: the points whose
label is in S, and each member's prediction restricted to S (the argmax of its logits
over S, with its softmax over S as the confidence).  Every independent pool contributes
its full problem and, for benchmarks with more than two classes, random class subsets of
a few sizes.  Per cell: the pooled collision probability, the best member's share of the
points a minority gets right, and the best committee of at most 40 members found by
greedy selection with the labels, with its gain over the full pool (plurality vote,
accuracy; Proposition 11).

    python -m analysis.regime_map
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from . import geometry, method, proposers
from .committees import Scorer, VoteAgg
from .pools import Pool, device, paper_pool
from .run import OUT, POOLS, clean, log, selection_gain

SUBSET_SIZES = {"kather": (2, 3, 5), "iwildcam": (2, 5, 10, 20, 50), "rxrx1": (20, 50, 200)}
N_DRAWS = 3
MIN_POINTS = 400          # points per cell
MIN_CLASS_POINTS = 10     # test points a class needs to enter a subset


def draw_subsets(y, K, sizes, rng):
    counts = np.bincount(y, minlength=K)
    eligible = np.flatnonzero(counts >= MIN_CLASS_POINTS)
    cells = []
    for s in sizes:
        if s >= len(eligible):
            continue
        got = 0
        for _ in range(500):
            S = np.sort(rng.choice(eligible, s, replace=False))
            if counts[S].sum() >= MIN_POINTS:
                cells.append(S)
                got += 1
                if got == N_DRAWS:
                    break
    return cells


def restricted_votes(pool, split, cells, y, dev):
    """Each member's top-1 label and confidence within each cell's classes."""
    rows = [np.flatnonzero(np.isin(y, S)) for S in cells]
    rows_t = [torch.as_tensor(r, device=dev) for r in rows]
    cols_t = [torch.as_tensor(S, device=dev) for S in cells]
    T = [np.empty((pool.M, len(r)), dtype=np.int16) for r in rows]
    C = [np.empty((pool.M, len(r)), dtype=np.float16) for r in rows]
    for i, mid in enumerate(pool.ids):
        lg = torch.from_numpy(np.load(pool.logit_path(split, mid)).astype(np.float32)).to(dev)
        for c in range(len(cells)):
            p = torch.softmax(lg[rows_t[c]][:, cols_t[c]], dim=1)
            conf, top = p.max(dim=1)
            T[c][i] = top.cpu().numpy().astype(np.int16)
            C[c][i] = conf.cpu().numpy().astype(np.float16)
        del lg
    return rows, T, C


def cell_terms(T, C, y_loc, K, dev, halves):
    """In-sample and cross-fitted regime terms of one cell (plurality vote, accuracy).

    Cross-fitted: the best committee (greedy, at most 40 members) and the best member
    (largest share of the minority-right points) are chosen on one half of the cell's
    points and evaluated on the other, both ways."""
    scorer = Scorer(y_loc, K, dev)
    col = geometry.collision_stats(T, y_loc, K)
    sg = selection_gain(T, C, y_loc, K, scorer)
    cf = []
    for fit, ev in halves:
        f, e = (torch.as_tensor(r, device=dev) for r in (fit, ev))
        sg_f = selection_gain(T[:, f], C[:, f], y_loc[fit], K, Scorer(y_loc[fit], K, dev))
        Te, Ce = T[:, e], C[:, e]
        sc_e = Scorer(y_loc[ev], K, dev)
        z_e = proposers.pool_plurality(Te, Ce, K)
        y_e = torch.as_tensor(y_loc[ev], device=dev)
        correct = Te == y_e[None]
        H = z_e != y_e
        headroom = float((H & correct.any(dim=0)).float().mean())
        agg = VoteAgg(Te, Ce, K)
        for j in sg_f["committee"]:
            agg.add(j)
        gain = sc_e(agg.pred, "acc") - sc_e(z_e, "acc")
        # the best member on the fit half, by its share of the minority-right points there
        f_sel = sg_f["best_member_index"]
        sg_e = selection_gain(Te, Ce, y_loc[ev], K, sc_e, z=z_e, kmax=1)
        share = sg_e["member_cover_minority"][f_sel] if sg_e["member_cover_minority"] else None
        cf.append({"k": sg_f["k"], "gain": gain, "headroom": headroom, "share": share})
    ok = all(c["share"] is not None and c["headroom"] > 0 for c in cf)
    return {"q": col["q"], "member_error": col["mean_member_error"],
            "best_member_cover_minority": sg["best_member_cover_minority"],
            "minority_right_mass": sg["minority_right_mass"], "headroom": sg["headroom"],
            "k": sg["k"], "gain": sg["realized_gain"], "full_acc": 1.0 - sg["full_err"],
            "cf_k": [c["k"] for c in cf],
            "cf_share": float(np.mean([c["share"] for c in cf])) if ok else None,
            "cf_recovered": float(np.mean([c["gain"] / c["headroom"] for c in cf])) if ok else None}


def analyse_pool(pool, seed=0):
    dev = device()
    split = pool.test
    y = pool.labels(split)
    dom = pool.domains(split)
    rng = np.random.default_rng(seed)
    cells = [np.arange(pool.K)] + draw_subsets(y, pool.K, SUBSET_SIZES.get(pool.benchmark, ()), rng)
    rows, Ts, Cs = restricted_votes(pool, split, cells, y, dev)
    cp = pool.compact(split, log=log)
    T_full = cp["top1"].astype(np.int64)
    C_full = cp["conf"].astype(np.float32)
    out = []
    for S, r, T, C in zip(cells, rows, Ts, Cs):
        K = len(S)
        y_loc = np.searchsorted(S, y[r])
        A, B, _ = method.domain_halves(dom[r], seed=seed)
        halves = ((A, B), (B, A))
        Tt = torch.as_tensor(T.astype(np.int64), device=dev)
        Ct = torch.as_tensor(C.astype(np.float32), device=dev)
        row = {"pool": pool.name, "benchmark": pool.benchmark, "K_full": pool.K,
               "cell": "full" if K == pool.K else K, "K": K, "classes": S.tolist(),
               "n": int(len(r))}
        row.update(cell_terms(Tt, Ct, y_loc, K, dev, halves))
        if K < pool.K:
            # sensitivity: a member whose unrestricted vote lies outside S keeps that vote,
            # which is wrong, instead of being restricted to S
            remap = np.full(pool.K, -1, dtype=np.int64)
            remap[S] = np.arange(K)
            outside = np.setdiff1d(np.arange(pool.K), S)
            remap[outside] = K + np.arange(len(outside))
            Ta = torch.as_tensor(remap[T_full[:, r]], device=dev)
            Ca = torch.as_tensor(C_full[:, r], device=dev)
            alt = cell_terms(Ta, Ca, y_loc, pool.K, dev, halves)
            row["alt"] = {k: alt[k] for k in ("q", "best_member_cover_minority", "k", "gain",
                                               "headroom", "cf_k", "cf_share", "cf_recovered")}
            del Ta, Ca
        out.append(row)
        del Tt, Ct
        torch.cuda.empty_cache()
    return out


def main(argv=None):
    roots = [os.path.join(POOLS, d) for d in sorted(os.listdir(POOLS))
             if paper_pool(d)
             and os.path.exists(os.path.join(POOLS, d, "predictions", "index.json"))]
    cells = []
    for root in roots:
        pool = Pool(root)
        if pool.config.get("mode", "build") != "build" or pool.M < pool.config.get("pool_size", 0):
            continue
        t0 = time.time()
        res = analyse_pool(pool)
        cells += res
        log(f"{pool.name}: {len(res)} cells  [{time.time() - t0:.0f}s]")
    path = os.path.join(OUT, "regime_map.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean({"cells": cells}), f)
    log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
