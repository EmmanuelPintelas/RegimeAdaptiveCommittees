"""Paired cluster bootstrap for the test-split comparisons.

Replicates resample the test split's domains with replacement (cameras on iWildCam,
experiments on RxRx1) and then points within each drawn domain, separately within each of
the two halves used for cross-fitting; Camelyon17 and Kather, whose test split is one
hospital or one cohort, are resampled by points within each half.  The same
1,000 replicates serve every method, so differences between methods are paired, and
intervals are percentile intervals.  Committees selected with labels are held fixed:
the intervals describe test-sampling variation given the selection.

Writes results/main/bootstrap/<pool>.npz (replicate scores per method) and
results/main/bootstrap.json (point scores, method names).

    python -m analysis.bootstrap --all
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from . import method
from .committees import VoteAgg
from .pools import Pool, device, paper_pool
from .run import GPU_PROB_BUDGET, OUT, POOLS, Split, clean, log
from .selection import BUDGETS, weighted_avg_pred, weighted_vote_pred

REPLICATES = 1000
CHUNK = 200


def replicate_weights(domains, B=REPLICATES, seed=0):
    """(B, n) multinomial weights, stratified by the two cross-fitting halves of the test
    split: within each half, its domains with replacement, then points within domains
    (points alone when the split is one domain), so that every replicate keeps both halves."""
    rng = np.random.default_rng(seed)
    d = np.asarray(domains)
    A, B_, _ = method.domain_halves(d, seed=0)
    n = len(d)
    W = np.zeros((B, n), dtype=np.float32)
    for stratum in (A, B_):
        u, inv = np.unique(d[stratum], return_inverse=True)
        if len(u) < 2:
            m = len(stratum)
            for b in range(B):
                W[b, stratum] = rng.multinomial(m, np.full(m, 1.0 / m))
            continue
        members = [stratum[inv == i] for i in range(len(u))]
        for b in range(B):
            draws = rng.multinomial(len(u), np.full(len(u), 1.0 / len(u)))
            for i in np.flatnonzero(draws):
                rows = members[i]
                W[b, rows] = rng.multinomial(draws[i] * len(rows),
                                             np.full(len(rows), 1.0 / len(rows)))
    return W


def batch_metric(pred, y, K, W, metric):
    """The metric under each row of weights W (B, n); classes present in each replicate."""
    Bn = W.shape[0]
    correct = pred == y
    if metric == "acc":
        return (W * correct).sum(dim=1) / W.sum(dim=1)
    N = torch.zeros((Bn, K), device=W.device).index_add_(1, y, W)
    TP = torch.zeros((Bn, K), device=W.device).index_add_(1, y[correct], W[:, correct])
    present = (N > 0).float()
    rec = TP / N.clamp_min(1e-12)
    cnt = present.sum(dim=1)
    if metric == "balanced_acc":
        return (rec * present).sum(dim=1) / cnt
    if metric == "gm":
        return torch.exp((torch.log(rec.clamp_min(1e-6)) * present).sum(dim=1) / cnt)
    if metric == "macro_f1":
        Pc = torch.zeros((Bn, K), device=W.device).index_add_(1, pred, W)
        prec = torch.where(Pc > 0, TP / Pc.clamp_min(1e-12), torch.zeros_like(TP))
        s = prec + rec
        f1 = torch.where(s > 0, 2 * prec * rec / s.clamp_min(1e-12), torch.zeros_like(s))
        return (f1 * present).sum(dim=1) / cnt
    raise ValueError(metric)


def replicate_scores(parts, y, K, W, metric, dev):
    """Scores of an estimator made of parts [(pred, rows)], averaged over parts, per replicate."""
    out = np.zeros(W.shape[0])
    for pred, rows in parts:
        r = torch.as_tensor(rows, device=dev) if rows is not None else None
        yy = y if r is None else y[r]
        pp = pred if r is None else pred[r]
        for s in range(0, W.shape[0], CHUNK):
            Wc = torch.as_tensor(W[s:s + CHUNK], device=dev)
            Wc = Wc if r is None else Wc[:, r]
            out[s:s + CHUNK] += batch_metric(pp, yy, K, Wc, metric).cpu().numpy() / len(parts)
    return out


def point_score(parts, y, K, metric, dev):
    ones = np.ones((1, len(y)), dtype=np.float32)
    return float(replicate_scores(parts, y, K, ones, metric, dev)[0])


def analyse_pool(pool, r, cf):
    dev = device()
    sp = Split(pool, pool.test, dev)
    y = torch.as_tensor(sp.y, device=dev)
    met, M, K = pool.metric, pool.M, pool.K
    on_gpu = pool.probs_bytes(pool.test) <= GPU_PROB_BUDGET
    P = pool.probs(pool.test, where="cuda" if on_gpu else "cpu")

    def avg_pred(members=None, weights=None):
        w = np.zeros(M)
        if weights is not None:
            w = np.asarray(weights, dtype=float)
        else:
            w[list(members)] = 1.0
        return weighted_avg_pred(P, w, dev)

    def vote_pred(members):
        agg = VoteAgg(sp.T, sp.C, K)
        for j in members:
            agg.add(j)
        return agg.pred.clone()

    preds = {"full pool": [(avg_pred(range(M)), None)]}
    if "avg" in r["method"]:
        A = r["method"]["avg"]
        preds["decider"] = [(avg_pred(A["main"]["members"]), None)]
        if A["main"].get("conservative"):
            preds["decider (conservative)"] = [(avg_pred(A["main"]["conservative"]["members"]),
                                                None)]
        preds["decider (uniform prior)"] = [(avg_pred(A["uniform_prior"]["members"]), None)]
        for name, d in A["ablations"].items():
            preds[f"ablation: {name}"] = [(avg_pred(d["members"]), None)]
        for b, d in A["budgets"].items():
            preds[f"budget {b}: decider"] = [(avg_pred(d["members"]), None)]
        Bl = r["baselines"]
        for name in ("best member on V", "top-k on V", "greedy on V"):
            preds[name] = [(avg_pred(Bl[name]["members"]), None)]
        preds["Caruana (bagged)"] = [(avg_pred(weights=Bl["Caruana (bagged)"]["weights"]), None)]
        preds["linear-pool stacking"] = [(avg_pred(weights=Bl["linear-pool stacking"]["weights"]),
                                          None)]
        preds["log-odds weighted vote"] = [
            (weighted_vote_pred(sp.T, np.asarray(Bl["log-odds weighted vote"]["weights"]), K), None)]
        paths = r["proposers"]["paths"]
        for crit in ("NLL", "BRIER"):
            d = Bl[f"proposed paths by {crit}"]
            preds[f"proposed paths by {crit}"] = [(avg_pred(paths[d["source"]][:d["k"]]), None)]
        d = Bl["proposed paths by balanced accuracy"]
        preds["proposed paths by balanced accuracy"] = [
            (avg_pred(paths[d["source"]][:d["k"]]), None)]
        for b in BUDGETS:
            row = Bl["budget"][str(b)] if str(b) in Bl["budget"] else Bl["budget"][b]
            preds[f"budget {b}: Caruana (capped)"] = [(avg_pred(weights=row["Caruana (capped) weights"]),
                                                       None)]
            for name, order in (("source-validation top-k", paths["P6 source val"]),
                                ("confidence top-k", paths["P1 confidence"]),
                                ("top-k on V", A["refit_orders"]["held-out top-k"]),
                                ("greedy on V", A["refit_orders"]["held-out greedy"])):
                preds[f"budget {b}: {name}"] = [(avg_pred(order[:b]), None)]
            for n in ("P7 collision-aware",) + tuple(p for p in paths if p.startswith(("P11", "P12"))):
                preds[f"budget {b}: {n}"] = [(avg_pred(paths[n][:b]), None)]
    if "vote" in r["method"]:
        V = r["method"]["vote"]
        preds["vote: decider"] = [(vote_pred(V["main"]["members"]), None)]
        preds["vote: full pool"] = [(vote_pred(range(M)), None)]
    # cross-fitted references (averaging): each half's committee scored on the other half
    if cf is not None:
        A_, B_, _ = method.domain_halves(pool.domains(pool.test), seed=0)
        evals = (B_, A_)
        g_parts, t_parts, f_parts = [], [], []
        for d, ev in zip(cf["avg"]["directions"], evals):
            g_parts.append((avg_pred(d["greedy_members"]), ev))
            t_parts.append((avg_pred(d["topk_members"]), ev))
            f_parts.append((preds["full pool"][0][0], ev))
        preds["cross-fitted greedy reference"] = g_parts
        preds["cross-fitted top-k at its size"] = t_parts
        preds["full pool (halves)"] = f_parts
    W = replicate_weights(pool.domains(pool.test))
    reps, points = {}, {}
    for name, parts in preds.items():
        reps[name] = replicate_scores(parts, y, K, W, met, dev).astype(np.float32)
        points[name] = point_score(parts, y, K, met, dev)
    del P
    torch.cuda.empty_cache()
    return reps, points


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    names = sorted(d for d in os.listdir(POOLS) if paper_pool(d) and "refined" not in d)
    cf_all = {}
    cf_path = os.path.join(OUT, "crossfit.json")
    if os.path.exists(cf_path):
        cf_all = json.load(open(cf_path, encoding="utf-8"))
    os.makedirs(os.path.join(OUT, "bootstrap"), exist_ok=True)
    idx_path = os.path.join(OUT, "bootstrap.json")
    index = json.load(open(idx_path, encoding="utf-8")) if os.path.exists(idx_path) else {}
    for name in names:
        if name in index and "--redo" not in argv:
            continue
        pool = Pool(os.path.join(POOLS, name))
        r = json.load(open(os.path.join(OUT, f"{name}.json"), encoding="utf-8"))
        t0 = time.time()
        reps, points = analyse_pool(pool, r, cf_all.get(name))
        np.savez_compressed(os.path.join(OUT, "bootstrap", f"{name}.npz"),
                            **{k.replace("/", "_"): v for k, v in reps.items()})
        index[name] = {"metric": pool.metric, "points": points,
                       "keys": {k: k.replace("/", "_") for k in reps}}
        with open(idx_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(clean(index), f)
        os.replace(idx_path + ".tmp", idx_path)
        log(f"{name}: {len(reps)} estimators  [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
