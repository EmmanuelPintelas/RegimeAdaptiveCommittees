"""The frozen evaluation: Algorithm 1 and every analysis of the paper, unchanged, on pools
built after the method was fixed (replicas of two reference pools, and pools of public
checkpoints).  Results go to results/frozen, never into the main results.

    python -m analysis.frozen public_clip_soups public_timm_in1k
    python -m analysis.frozen camelyon17_seed_4 rxrx1_seed_4 ...
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from . import bootstrap, crossfit, run
from .pools import CODE, Pool, paper_pool
from .run import POOLS, clean, log

FROZEN = os.path.join(CODE, "results", "frozen")


def point_to_frozen():
    """Every module that reads or writes results under OUT works under FROZEN instead."""
    for mod in (run, crossfit, bootstrap):
        mod.OUT = FROZEN


def evaluate(name):
    if paper_pool(name):
        raise SystemExit(f"{name} is a pool of the main analyses, not of the frozen evaluation")
    pool = Pool(os.path.join(POOLS, name))
    os.makedirs(os.path.join(FROZEN, "bootstrap"), exist_ok=True)
    path = os.path.join(FROZEN, f"{name}.json")
    if not os.path.exists(path):
        log(f"\n=== {name}: {pool.benchmark}, K={pool.K}, M={pool.M}, metric {pool.metric}")
        res = run.analyse(pool)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(clean(res), f)
        os.replace(path + ".tmp", path)
        log(f"  analysis [{res['seconds'] / 60:.1f} min]")
    r = json.load(open(path, encoding="utf-8"))
    cf_path = os.path.join(FROZEN, "crossfit.json")
    cf = json.load(open(cf_path, encoding="utf-8")) if os.path.exists(cf_path) else {}
    if name not in cf:
        t0 = time.time()
        cf[name] = crossfit.analyse_pool(pool)
        with open(cf_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(clean(cf), f)
        os.replace(cf_path + ".tmp", cf_path)
        log(f"  cross-fitting [{(time.time() - t0) / 60:.1f} min]")
    bs_path = os.path.join(FROZEN, "bootstrap.json")
    index = json.load(open(bs_path, encoding="utf-8")) if os.path.exists(bs_path) else {}
    if name not in index:
        t0 = time.time()
        reps, points = bootstrap.analyse_pool(pool, r, cf[name])
        np.savez_compressed(os.path.join(FROZEN, "bootstrap", f"{name}.npz"),
                            **{k.replace("/", "_"): v for k, v in reps.items()})
        index[name] = {"metric": pool.metric, "points": points,
                       "keys": {k: k.replace("/", "_") for k in reps}}
        with open(bs_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(clean(index), f)
        os.replace(bs_path + ".tmp", bs_path)
        log(f"  bootstrap [{(time.time() - t0) / 60:.1f} min]")


LABELS = {"public_clip_soups": "CLIP soups", "public_timm_in1k": "timm ImageNet-1k",
          "camelyon17_seed_4": "Camelyon17, seed 4", "camelyon17_seed_5": "Camelyon17, seed 5",
          "rxrx1_seed_4": "RxRx1, seed 4", "rxrx1_seed_5": "RxRx1, seed 5"}


def table(names):
    """paper/generated/tab_frozen.tex: one row per frozen pool."""
    from . import report as R
    R.RESULTS = FROZEN
    R.CF, R.BS = R.load_json("crossfit.json"), R.load_json("bootstrap.json")
    body = []
    for name in names:
        path = os.path.join(FROZEN, f"{name}.json")
        if not os.path.exists(path):
            continue
        r = json.load(open(path, encoding="utf-8"))
        c, m = R.cf(r), r["method"]["avg"]
        q = [r["splits"][s]["collision"]["q"] for s in (r["tuning"], r["test"])]
        tr = m["main"]["transfer"]
        two = R.diff([r], "ablation: full pool vs held-out greedy", "full pool")
        body.append(" & ".join([
            LABELS.get(name, name), str(r["K"]), f"{q[0]:.2f}/{q[1]:.2f}",
            f"{r['label_free']['q_free']:.2f}", R.ks(c["avg"]["k"]),
            R.fmt_ci(R.diff([r], "cross-fitted greedy reference", "full pool (halves)")),
            f"{c['gain']['best_member_share']:.2f}", f"{c['certificate']['B']:.2f}",
            f"{R.short_source(m['main']['source'])} {m['main']['k']}",
            R.fmt_ci(R.diff([r], "decider", "full pool")), R.signed(R.point(r, "ablation: full "
                     "pool vs held-out greedy") - R.point(r, "full pool")) if two else "--",
            f"{tr['gamma']:.2f}", f"{tr['dis_T']:.2f}"]) + r" \\")
    R.table("frozen", r"Frozen evaluation, test split, probability averaging, geometric-mean "
            r"recall for $K<11$ and macro-F1 for $K\ge11$. $\coll$: tuning/test; $\tilde\coll$: "
            r"label-free. Greedy reference (test labels): cross-fitted sizes and gain over the "
            r"full pool; best member: its cross-fitted share of minority-right points; $B(k)$: "
            r"Proposition~\ref{prop:certificate}. Decider: Algorithm~\ref{alg:select}, "
            r"unchanged, its choice and difference from the full pool; two-cand.: full pool against held-out "
            r"greedy; $\gamma$, $\mathrm{dis}_{\mathrm T}$: as in Table~\ref{tab:gamma}.",
            "lr" + "c" * 11,
            r"pool & $K$ & $\coll$ & $\tilde\coll$ & $k$ & greedy ref. & best member & $B(k)$ "
            r"& decider & $-$ full & two-cand. & $\gamma$ & $\mathrm{dis}_{\mathrm T}$",
            body, tabcolsep="2.5pt")


ORDER = ("public_clip_soups", "public_timm_in1k", "camelyon17_seed_4", "camelyon17_seed_5",
         "rxrx1_seed_4", "rxrx1_seed_5")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    point_to_frozen()
    if argv == ["--table"]:
        table(ORDER)
        return 0
    for name in argv:
        evaluate(name)
    table(ORDER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
