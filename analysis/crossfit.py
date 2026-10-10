"""Oracle references done honestly: cross-fitted, checked by stronger search, certified.

Cross-fitting.  The test split is halved by domain (cameras on iWildCam, experiments on
RxRx1; random halves on Camelyon17 and Kather, whose test split is one hospital or one
cohort).  Every quantity selected with test labels (a greedy path and its size, a
top-k ordering, the best member, the P7 oracle plug-ins) is selected on one half and
evaluated on the other, both ways; the two directions are averaged.  The in-sample
value, selected and evaluated on the whole test split, is kept as the optimistic
reference.

Stronger search.  For sizes k in {greedy-reference size, 5, 10, 20, 40}: twenty
greedy runs from a random first member, swap local search from the greedy committee,
and backward elimination from the full pool, all in-sample under probability
averaging; the best score any of them finds is compared with greedy's.

Certificate.  B(k) of Proposition 8', the sum of the k largest
cov_i(k) = E[1{x in H, member i right} / ceil(k / (D(x) + 1))], D(x) the number of
distinct wrong labels the pool casts at x; under the deployed vote rule.

    python -m analysis.crossfit --all
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from . import geometry, method, proposers
from .committees import (AvgAgg, RowAvgAgg, Scorer, VoteAgg, greedy, member_scores,
                         path_scores)
from .pools import Pool, device, paper_pool
from .run import GPU_PROB_BUDGET, GREEDY_SUBSAMPLE, OUT, POOLS, Split, clean, log, nu_bounds

SEARCH_SIZES = (5, 10, 20, 40)
RANDOM_STARTS = 20
KMAX_GAIN = 40


class SumAgg:
    """A committee kept as a dense score sum, so that members can also be removed."""

    def __init__(self, contrib, N, K, dev):
        self.contrib = contrib            # j -> (N, K) dense contribution on dev
        self.N, self.K, self.dev = N, K, dev
        self.reset()

    def reset(self):
        self.S = torch.zeros((self.N, self.K), dtype=torch.float32, device=self.dev)
        self.members = []

    @property
    def pred(self):
        return self.S.argmax(dim=1)

    def add(self, j):
        self.S += self.contrib(j)
        self.members.append(int(j))

    def remove(self, j):
        self.S -= self.contrib(j)
        self.members.remove(int(j))

    def trial(self, j):
        return (self.S + self.contrib(j)).argmax(dim=1)

    def trial_remove(self, j):
        return (self.S - self.contrib(j)).argmax(dim=1)

    def trial_swap(self, out, inn):
        return (self.S - self.contrib(out) + self.contrib(inn)).argmax(dim=1)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def best_k(curve):
    return int(np.argmax(curve)) + 1


def avg_agg(P, rows, dev, on_gpu):
    """Averaging aggregator on the given rows (a subsample of them for selection when P is
    in host memory, as for RxRx1)."""
    return RowAvgAgg(P, rows, dev)


def selection_rows(rows, on_gpu, seed=0):
    if on_gpu or len(rows) <= GREEDY_SUBSAMPLE:
        return np.asarray(rows)
    return np.sort(np.random.default_rng(seed).choice(rows, GREEDY_SUBSAMPLE, replace=False))


def coverage_terms(T, y, z, K):
    """Per-point H, per-member correctness on H, minority-right mask, D(x)."""
    y_t = torch.as_tensor(np.asarray(y), device=T.device)
    correct = T == y_t[None]
    H = z != y_t
    M = T.shape[0]
    w = (~correct).sum(dim=0)
    p_pt = 1.0 - w.float() / M
    cdis = geometry.disagreement_counts(T, y, K).float()
    pimax = torch.where(w > 0, cdis.max(dim=1).values / w.clamp_min(1).float(),
                        torch.zeros_like(p_pt))
    minority = ((p_pt - (1 - p_pt) * pimax) < 0) & (p_pt > 0)
    D = (cdis > 0).sum(dim=1)
    del cdis
    return correct, H, minority, D


def certificate(correct, H, D, k):
    """B(k): the sum of the k largest cov_i(k) (Proposition 8')."""
    nu = torch.ceil(k / (D.float() + 1.0)).clamp_min(1.0)
    cov_k = ((correct & H[None]).float() / nu[None]).mean(dim=1)
    return float(torch.topk(cov_k, k).values.sum())


# --------------------------------------------------------------------------- #
# cross-fitting
# --------------------------------------------------------------------------- #
def crossfit_avg(pool, sp, P, halves, dev, on_gpu):
    """Greedy reference and top-k under averaging, selected on one half, scored on the other."""
    metric = pool.metric
    dirs = []
    for fit, ev in halves:
        sel = selection_rows(fit, on_gpu)
        sc_sel = Scorer(sp.y[sel], pool.K, dev)
        if on_gpu:
            g_order, g_fit = greedy(avg_agg(P, sel, dev, on_gpu), sc_sel, metric)
        else:
            P_sel = P[:, torch.as_tensor(sel)].to(dev)
            g_order, g_fit = greedy(AvgAgg(P_sel, dev), sc_sel, metric)
            del P_sel
            torch.cuda.empty_cache()
        sc_fit = Scorer(sp.y[fit], pool.K, dev)
        ms = member_scores(sp.T[:, torch.as_tensor(fit, device=dev)], sc_fit, metric)
        t_order = [int(j) for j in np.argsort(-ms, kind="stable")]
        sc_ev = Scorer(sp.y[ev], pool.K, dev)
        agg_ev = avg_agg(P, ev, dev, on_gpu)
        g_ev = path_scores(agg_ev, g_order, sc_ev, (metric,))[metric]
        t_ev = path_scores(agg_ev, t_order, sc_ev, (metric,))[metric]
        k = best_k(g_fit)
        dirs.append({"k": k, "greedy_ev": g_ev, "topk_ev": t_ev, "full_ev": g_ev[-1],
                     "score": g_ev[k - 1], "gap": g_ev[k - 1] - t_ev[k - 1],
                     "best_member_ev": t_ev[0], "greedy_members": g_order[:k],
                     "topk_members": t_order[:k]})
        torch.cuda.empty_cache()
    mean = lambda key: float(np.mean([d[key] for d in dirs]))
    return {"directions": dirs, "k": [d["k"] for d in dirs], "score": mean("score"),
            "full": mean("full_ev"), "gap": mean("gap"),
            "greedy_curve": np.mean([d["greedy_ev"] for d in dirs], axis=0).tolist(),
            "topk_curve": np.mean([d["topk_ev"] for d in dirs], axis=0).tolist()}


def crossfit_vote(pool, sp, halves, dev, weighted=False, metric=None, kmax=None):
    """Greedy reference under the (weighted) vote, selected on one half, scored on the other."""
    metric = metric or pool.metric
    dirs = []
    for fit, ev in halves:
        f, e = (torch.as_tensor(r, device=dev) for r in (fit, ev))
        g_order, g_fit = greedy(VoteAgg(sp.T[:, f], sp.C[:, f], pool.K, weighted=weighted),
                                Scorer(sp.y[fit], pool.K, dev), metric, kmax=kmax)
        k = best_k(g_fit)
        sc_ev = Scorer(sp.y[ev], pool.K, dev)
        agg_ev = VoteAgg(sp.T[:, e], sp.C[:, e], pool.K, weighted=weighted)
        g_ev = path_scores(agg_ev, g_order, sc_ev, (metric,))[metric]
        full = path_scores(agg_ev, list(range(pool.M)), sc_ev, (metric,))[metric][-1]
        dirs.append({"k": k, "score": g_ev[k - 1], "full_ev": full, "order": g_order[:k]})
    return {"directions": dirs, "k": [d["k"] for d in dirs],
            "score": float(np.mean([d["score"] for d in dirs])),
            "full": float(np.mean([d["full_ev"] for d in dirs]))}


def crossfit_gain(pool, sp, halves, dev):
    """Table 5 cross-fitted: greedy (vote, accuracy, at most 40 members) on one half,
    its gain over the full pool, coverage terms and bounds on the other."""
    dirs = []
    for fit, ev in halves:
        f, e = (torch.as_tensor(r, device=dev) for r in (fit, ev))
        Tf, Cf, Te, Ce = sp.T[:, f], sp.C[:, f], sp.T[:, e], sp.C[:, e]
        order, curve = greedy(VoteAgg(Tf, Cf, pool.K), Scorer(sp.y[fit], pool.K, dev), "acc",
                              kmax=KMAX_GAIN)
        k = best_k(curve)
        comm = order[:k]
        z_f = proposers.pool_plurality(Tf, Cf, pool.K)
        corr_f, H_f, minor_f, _ = coverage_terms(Tf, sp.y[fit], z_f, pool.K)
        best = int(torch.argmax(corr_f[:, minor_f].float().mean(dim=1))) if bool(minor_f.any()) else 0
        z_e = proposers.pool_plurality(Te, Ce, pool.K)
        corr_e, H_e, minor_e, D_e = coverage_terms(Te, sp.y[ev], z_e, pool.K)
        sc_e = Scorer(sp.y[ev], pool.K, dev)
        agg = VoteAgg(Te, Ce, pool.K)
        acc_c = path_scores(agg, comm, sc_e, ("acc",))["acc"][-1]
        full_acc = sc_e(z_e, "acc")
        cov_e = (corr_e & H_e[None]).float().mean(dim=1)
        nu_ge, _ = nu_bounds(k, pool.K)
        dirs.append({
            "k": k, "gain": acc_c - full_acc,
            "bound": float(cov_e[comm].sum()) / nu_ge,
            "certificate": certificate(corr_e, H_e, D_e, k),
            "best_member_share": (float(corr_e[best, minor_e].float().mean())
                                  if bool(minor_e.any()) else None),
            "headroom": float((H_e & corr_e.any(dim=0)).float().mean())})
    keys = ("gain", "bound", "certificate", "best_member_share", "headroom")
    return {"directions": dirs, "k": [d["k"] for d in dirs],
            **{key: (float(np.mean([d[key] for d in dirs]))
                     if all(d[key] is not None for d in dirs) else None) for key in keys}}


def crossfit_p7_oracle(pool, sp, halves, dev):
    """P7 with test correctness as plug-ins: built on one half, scored on the other."""
    dirs = []
    for fit, ev in halves:
        f, e = (torch.as_tensor(r, device=dev) for r in (fit, ev))
        y_f = torch.as_tensor(sp.y[fit], device=dev)
        correct = (sp.T[:, f] == y_f[None]).to(torch.uint8)
        q = geometry.collision_stats(sp.T[:, f], sp.y[fit], pool.K)["q"]
        L = max(1, round(1 / q)) if q and q > 0 else 1
        order, _, stop = proposers.model_greedy(correct[:, None, :],
                                                proposers.model_table(pool.M, L, dev))
        sc_e = Scorer(sp.y[ev], pool.K, dev)
        agg = VoteAgg(sp.T[:, e], sp.C[:, e], pool.K)
        dirs.append({"k": int(stop),
                     "score": path_scores(agg, order[:stop], sc_e, (pool.metric,))[pool.metric][-1]})
    return {"directions": dirs, "k": [d["k"] for d in dirs],
            "score": float(np.mean([d["score"] for d in dirs]))}


# --------------------------------------------------------------------------- #
# stronger search (in-sample, averaging)
# --------------------------------------------------------------------------- #
def stronger_search(pool, sp, P, dev, on_gpu, k_ref, g_order, seed=0):
    rows = selection_rows(np.arange(sp.n), on_gpu, seed)
    r_t = torch.as_tensor(rows, device=P.device)
    sc = Scorer(sp.y[rows], pool.K, dev)
    metric = pool.metric
    cache = {}

    def contrib(j):
        if j not in cache:
            cache[j] = P[j][r_t].to(dev)
        return cache[j].float()

    agg = SumAgg(contrib, len(rows), pool.K, dev)
    sizes = sorted({int(k_ref), *SEARCH_SIZES})

    def score_of(members):
        agg.reset()
        for j in members:
            agg.add(j)
        return sc(agg.pred, metric)

    out = {k: {"greedy": score_of(g_order[:k])} for k in sizes}
    # (i) greedy from a random first member
    rng = np.random.default_rng(seed)
    best_rand = {k: -np.inf for k in sizes}
    for _ in range(RANDOM_STARTS):
        start = int(rng.integers(pool.M))
        agg.reset()
        agg.add(start)
        if 1 in best_rand:
            best_rand[1] = max(best_rand[1], sc(agg.pred, metric))
        remaining = [j for j in range(pool.M) if j != start]
        for size in range(2, max(sizes) + 1):
            s_best, j_best = -np.inf, None
            for j in remaining:
                s = sc(agg.trial(j), metric)
                if s > s_best:
                    s_best, j_best = s, j
            agg.add(j_best)
            remaining.remove(j_best)
            if size in best_rand:
                best_rand[size] = max(best_rand[size], s_best)
    # (ii) swap local search from the greedy committee
    best_swap = {}
    for k in sizes:
        comm = list(g_order[:k])
        cur = score_of(comm)
        for _ in range(20):
            improved = False
            out_set = [j for j in range(pool.M) if j not in comm]
            best_move = (cur, None, None)
            for a in comm:
                for b in out_set:
                    s = sc(agg.trial_swap(a, b), metric)
                    if s > best_move[0] + 1e-12:
                        best_move = (s, a, b)
            if best_move[1] is not None:
                agg.remove(best_move[1])
                agg.add(best_move[2])
                comm[comm.index(best_move[1])] = best_move[2]
                cur = best_move[0]
                improved = True
            if not improved:
                break
        best_swap[k] = cur
    # (iii) backward elimination from the full pool
    best_back = {}
    agg.reset()
    for j in range(pool.M):
        agg.add(j)
    comm = list(range(pool.M))
    while len(comm) > 1:
        s_best, j_best = -np.inf, None
        for j in comm:
            s = sc(agg.trial_remove(j), metric)
            if s > s_best:
                s_best, j_best = s, j
        agg.remove(j_best)
        comm.remove(j_best)
        if len(comm) in sizes:
            best_back[len(comm)] = s_best
    for k in sizes:
        out[k]["random_start"] = best_rand.get(k)
        out[k]["swap"] = best_swap[k]
        out[k]["backward"] = best_back.get(k)
        out[k]["best"] = max(v for v in (out[k]["greedy"], best_rand.get(k), best_swap[k],
                                         best_back.get(k)) if v is not None and v > -np.inf)
    cache.clear()
    return {"rows": len(rows), "k_ref": int(k_ref), "sizes": out}


# --------------------------------------------------------------------------- #
def analyse_pool(pool):
    dev = device()
    t0 = time.time()
    r = json.load(open(os.path.join(OUT, f"{pool.name}.json"), encoding="utf-8"))
    test = pool.test
    sp = Split(pool, test, dev)
    A, B, note = method.domain_halves(pool.domains(test))
    halves = ((A, B), (B, A))
    on_gpu = pool.probs_bytes(test) <= GPU_PROB_BUDGET
    P = pool.probs(test, where="cuda" if on_gpu else "cpu")
    res = {"pool": pool.name, "benchmark": pool.benchmark, "halves": note}
    res["avg"] = crossfit_avg(pool, sp, P, halves, dev, on_gpu)
    log(f"  avg cross-fitted [{time.time() - t0:.0f}s]")
    res["vote"] = crossfit_vote(pool, sp, halves, dev)
    res["wvote"] = crossfit_vote(pool, sp, halves, dev, weighted=True)
    res["gain"] = crossfit_gain(pool, sp, halves, dev)
    res["p7_oracle"] = crossfit_p7_oracle(pool, sp, halves, dev)
    log(f"  vote, gain, P7 oracle [{time.time() - t0:.0f}s]")
    # the certificate on the whole test split, at the in-sample greedy size
    z = proposers.pool_plurality(sp.T, sp.C, pool.K)
    corr, H, _, D = coverage_terms(sp.T, sp.y, z, pool.K)
    k_in = int(r["selection_gain"]["k"])
    res["certificate"] = {"k": k_in, "B": certificate(corr, H, D, k_in),
                          "B_by_k": {k: certificate(corr, H, D, k) for k in (1, 2, 5, 10, 20, 40)},
                          "mean_D_on_H": float(D[H].float().mean()) if bool(H.any()) else None}
    sw = r["sweep"]["avg"][test]
    g_curve = sw["curves"]["greedy"][pool.metric]
    res["search"] = stronger_search(pool, sp, P, dev, on_gpu, best_k(g_curve),
                                    sw["orders"]["greedy"])
    log(f"  stronger search [{time.time() - t0:.0f}s]")
    del P
    torch.cuda.empty_cache()
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    names = sorted(d for d in os.listdir(POOLS) if paper_pool(d) and "refined" not in d) \
        if "--all" in argv else [os.path.basename(a.rstrip("/\\")) for a in argv]
    path = os.path.join(OUT, "crossfit.json")
    done = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for name in names:
        if name in done:
            log(f"{name}: done")
            continue
        pool = Pool(os.path.join(POOLS, name))
        log(f"\n=== {name}")
        done[name] = analyse_pool(pool)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(clean(done), f)
        os.replace(path + ".tmp", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
