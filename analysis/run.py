"""Every analysis for one or more pools, written to results/main/<pool>.json.

    python -m analysis.run pools/camelyon17_seed_1 pools/rxrx1_seed_1
    python -m analysis.run --all

Per pool and per split (tuning, test):
    members      each member's score (pool metric) and balanced accuracy
    collision    pooled collision spectrum, regime masses (labels)
    random       random-committee zero-one error, measured and predicted
    sweep        committee score against size under vote, wvote and avg for the
                 rules random, confidence (label-free), top-k and greedy (labels)
Per pool:
    label_free   collision against the pool plurality, and the Prop. 20 bounds
    geometry     Brier ambiguity share and error alignment on the tuning split
    proposers    P1-P7 orderings from the unlabeled test inputs; P7 stopping size
    method       the decider on the tuning split under vote and avg, its terms,
                 baselines, and every proposer at its own tuned size

For RxRx1 the average of probabilities on the test split streams members from
host memory, and its greedy search runs on a fixed subsample of test points;
everything else uses every point.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from collections import Counter

import numpy as np
import torch

from . import geometry, method, predict, proposers, selection, weight_noise
from .committees import AvgAgg, Scorer, VoteAgg, greedy, member_scores, path_scores
from .pools import CODE, Pool, device, paper_pool

sys.path.insert(0, CODE)
import paths  # noqa: E402  (the repository root's paths.py)

OUT = os.path.join(CODE, "results", "main")
POOLS = paths.POOLS
GPU_PROB_BUDGET = 2.6 * 2 ** 30
GREEDY_SUBSAMPLE = 10000


def log(msg):
    print(msg, flush=True)


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if torch.is_tensor(obj):
        return clean(obj.cpu().numpy())
    return obj


class Split:
    def __init__(self, pool, name, dev):
        self.name = name
        cp = pool.compact(name, log=log)
        self.y = pool.labels(name)
        self.T = torch.as_tensor(cp["top1"].astype(np.int64), device=dev)
        self.C = torch.as_tensor(cp["conf"].astype(np.float32), device=dev)
        self.scorer = Scorer(self.y, pool.K, dev)
        self.n = len(self.y)


def rule_curves(agg, scorer, metric, orders, metrics):
    """{rule: {metric: curve}}; rules named random_* are averaged into 'random'."""
    out, rand = {}, []
    for rule, order in orders.items():
        c = path_scores(agg, order, scorer, metrics)
        if rule.startswith("random_"):
            rand.append(c)
        else:
            out[rule] = c
    if rand:
        out["random"] = {m: np.mean([c[m] for c in rand], axis=0).tolist() for m in metrics}
    return out


def vote_sweep(pool, sp, weighted, R, seed):
    rng = np.random.default_rng(seed)
    agg = VoteAgg(sp.T, sp.C, pool.K, weighted=weighted)
    ms = member_scores(sp.T, sp.scorer, pool.metric)
    orders = {f"random_{r}": list(rng.permutation(pool.M)) for r in range(R)}
    orders["confidence"] = list(np.argsort(-sp.C.mean(dim=1).cpu().numpy(), kind="stable"))
    orders["topk"] = list(np.argsort(-ms, kind="stable"))
    g_order, _ = greedy(agg, sp.scorer, pool.metric)
    orders["greedy"] = g_order
    curves = rule_curves(agg, sp.scorer, pool.metric, orders, (pool.metric, "balanced_acc", "acc"))
    return curves, {k: [int(v) for v in o] for k, o in orders.items() if not k.startswith("random_")}


def avg_sweep(pool, sp, P, R, seed, P_greedy=None, greedy_rows=None):
    dev = device()
    rng = np.random.default_rng(seed)
    agg = AvgAgg(P, dev)
    ms = member_scores(sp.T, sp.scorer, pool.metric)
    orders = {f"random_{r}": list(rng.permutation(pool.M)) for r in range(R)}
    orders["confidence"] = list(np.argsort(-sp.C.mean(dim=1).cpu().numpy(), kind="stable"))
    orders["topk"] = list(np.argsort(-ms, kind="stable"))
    if P_greedy is None:
        g_order, _ = greedy(agg, sp.scorer, pool.metric)
    else:
        sub_scorer = Scorer(sp.y[greedy_rows], pool.K, dev)
        g_order, _ = greedy(AvgAgg(P_greedy, dev), sub_scorer, pool.metric)
    orders["greedy"] = g_order
    curves = rule_curves(agg, sp.scorer, pool.metric, orders, (pool.metric, "balanced_acc", "acc"))
    return curves, {k: [int(v) for v in o] for k, o in orders.items() if not k.startswith("random_")}


def nu_bounds(k, K):
    """Fewest correct votes with which a plurality of k members can be right (Prop. 8).

    nu_ge = ceil(k / K) when a tie may go to the true label (every deployable rule);
    nu_gt = 1 + ceil((k - 1) / K) when ties count as errors."""
    return -(-k // K), 1 + (-(-(k - 1) // K))


def selection_gain(T, C, y, K, scorer, z=None, kmax=40):
    """What selection can gain over the full pool, in accuracy, under the plurality vote.

    Proposition 11's terms: the pool's mistakes H, each member's coverage of them, the
    points where the strongest wrong label out-polls the true one although some member is
    right (minority-right), and the best committee of at most ``kmax`` members found by
    greedy forward selection with the labels."""
    M = T.shape[0]
    y_t = torch.as_tensor(np.asarray(y), device=T.device)
    if z is None:
        z = proposers.pool_plurality(T, C, K)
    correct = T == y_t[None]
    H = z != y_t
    cov = (correct & H[None]).float().mean(dim=1)
    acc_order, acc_curve = greedy(VoteAgg(T, C, K), scorer, "acc", kmax=kmax)
    ka = int(np.argmax(acc_curve)) + 1
    nu_ge, nu_gt = nu_bounds(ka, K)
    full_acc = scorer(z, "acc")
    w = (T != y_t[None]).sum(dim=0)
    p_pt = 1.0 - w.float() / M
    cdis = geometry.disagreement_counts(T, y, K).float()
    pimax = torch.where(w > 0, cdis.max(dim=1).values / w.clamp_min(1).float(),
                        torch.zeros_like(p_pt))
    minority = ((p_pt - (1 - p_pt) * pimax) < 0) & (p_pt > 0)
    del cdis
    cover_minority = correct[:, minority].float().mean(dim=1) if bool(minority.any()) else cov * 0
    return {
        "full_err": 1.0 - full_acc,
        "headroom": float((H & correct.any(dim=0)).float().mean()),
        "best_member_cover": float(cov.max()), "k": ka, "committee": acc_order[:ka],
        "realized_gain": acc_curve[ka - 1] - full_acc,
        "bound": float(cov[acc_order[:ka]].sum()) / nu_ge,
        "bound_ties_as_errors": float(cov[acc_order[:ka]].sum()) / nu_gt,
        "nu_ge": nu_ge, "nu_gt": nu_gt,
        "minority_right_mass": float(minority.float().mean()),
        "best_member_cover_minority": float(cover_minority.max()),
        "mean_member_cover_minority": float(cover_minority.mean()),
        "best_member_index": int(torch.argmax(cover_minority)),
        "member_cover_minority": (cover_minority.cpu().numpy().tolist()
                                  if bool(minority.any()) else None)}


def analyse(pool, seed=0):
    dev = device()
    t0 = time.time()
    res = {"pool": pool.name, "benchmark": pool.benchmark, "K": pool.K, "M": pool.M,
           "mode": pool.config.get("mode", "build"),
           "metric": pool.metric, "archs": dict(Counter(pool.archs)),
           "schedule": dict(Counter(pool.schedule_modes())), "ids": pool.ids,
           "tuning": pool.tuning, "test": pool.test, "splits": {}, "sweep": {}, "method": {}}
    names = [s for s in (pool.tuning, pool.test) if s]
    sp = {s: Split(pool, s, dev) for s in names}
    R_pred = 400 if pool.K == 2 else 40
    R_vote = 20

    for s in names:
        d = sp[s]
        cs = geometry.collision_stats(d.T, d.y, pool.K)
        mc = predict.correct_counts(d.T, d.y)
        L_q = max(1, round(1 / cs["q"])) if cs["q"] and cs["q"] > 0 else 1
        res["splits"][s] = {
            "n": d.n, "role": pool.splits[s],
            "member_scores": member_scores(d.T, d.scorer, pool.metric),
            "member_balanced_acc": member_scores(d.T, d.scorer, "balanced_acc"),
            "member_acc": member_scores(d.T, d.scorer, "acc"),
            "collision": cs,
            "random_measured": predict.measured_random(d.T, d.y, pool.K, R=R_pred, seed=seed),
            "random_pred_q": predict.predicted_random(mc, d.y, pool.M, L_q, seed=seed),
            "random_pred_K": predict.predicted_random(mc, d.y, pool.M, pool.K - 1, seed=seed),
            "random_measured_ties_as_errors": predict.measured_random(
                d.T, d.y, pool.K, R=R_pred, seed=seed, tie="error"),
            "calibration": geometry.calibration(d.T, d.C, d.y),
            "mu_split": geometry.split_sample_mu(d.T, d.y, pool.K),
        }
        log(f"  {s}: collision q={cs['q']:.4f} q3={cs['q3']:.4f}, majority-wrong "
            f"{cs['frac_majority_wrong']:.3f}  [{time.time() - t0:.0f}s]")

    # label-free collision on the test split, against the pool plurality
    dt = sp[pool.test]
    paths, info = proposers.proposer_paths(pool, dt.T, dt.C)
    z = info["z"]
    q_labels = res["splits"][pool.test]["collision"]["q"]
    res["label_free"] = geometry.label_free_bounds(dt.T, z.cpu().numpy(), dt.y, pool.K,
                                                   q_labels, info["q_free"])
    dec_t = geometry.pair_decomposition(dt.T, z, dt.y, pool.K)
    res["label_free"]["decomposition_test"] = dec_t
    if pool.tuning:
        dv_ = sp[pool.tuning]
        z_v = proposers.pool_plurality(dv_.T, dv_.C, pool.K)
        dec_v = geometry.pair_decomposition(dv_.T, z_v, dv_.y, pool.K)
        res["label_free"]["decomposition_tuning"] = dec_v
        res["label_free"]["q_corrected"] = geometry.corrected_q(dec_t, dec_v)
    y_t = torch.as_tensor(dt.y, device=dev)
    correct = (dt.T == y_t[None]).to(torch.uint8)
    L_lab = max(1, round(1 / q_labels))
    o_order, o_err, o_stop = proposers.model_greedy(
        correct[:, None, :], proposers.model_table(pool.M, L_lab, dev))
    z_order, z_err, z_stop = proposers.model_greedy(
        (dt.T == z[None]).to(torch.uint8)[:, None, :],
        proposers.model_table(pool.M, info["L_free"], dev))
    gap = (dt.C - correct.float()).abs().mean(dim=1).cpu().numpy()      # E|conf_i - c_i|
    p7 = paths["P7 collision-aware"]
    res["label_free"]["plugin_bound"] = float(2 * gap[p7[:info["p7_stop"]]].sum())
    res["label_free"]["mean_conf_gap"] = float(gap.mean())
    res["proposers"] = {"paths": paths, "p7_stop": info["p7_stop"],
                        "p7_model_errors": info["p7_model_errors"], "q_free": info["q_free"],
                        "L_free": info["L_free"], "L_labels": L_lab,
                        "p7_oracle_path": o_order, "p7_oracle_stop": o_stop,
                        "p7_oracle_model_errors": o_err,
                        "p7_z_path": z_order, "p7_z_stop": z_stop}
    log(f"  proposers: {len(paths)} paths; P7 size {info['p7_stop']} (oracle {o_stop}, "
        f"pool-agreement {z_stop}); q free {info['q_free']:.4f} vs labels {q_labels:.4f}  "
        f"[{time.time() - t0:.0f}s]")

    extra = {"P7 oracle": o_order, "P7 pool-agreement": z_order}
    vt = VoteAgg(dt.T, dt.C, pool.K)
    res["p7_test_curves"] = {
        n: path_scores(vt, o, dt.scorer, (pool.metric,))[pool.metric]
        for n, o in {"P7 collision-aware": paths["P7 collision-aware"], **extra}.items()}

    # what selection can gain, in accuracy, under the plurality vote (Proposition 11)
    res["selection_gain"] = selection_gain(dt.T, dt.C, dt.y, pool.K, dt.scorer, z=z)
    for agg_name in ("vote", "wvote"):
        res["sweep"][agg_name] = {}
        for s in names:
            curves, orders = vote_sweep(pool, sp[s], agg_name == "wvote", R_vote, seed)
            res["sweep"][agg_name][s] = {"curves": curves, "orders": orders}
        log(f"  sweep {agg_name} done  [{time.time() - t0:.0f}s]")
    dom_v = pool.domains(pool.tuning) if pool.tuning else None
    if pool.tuning:
        dv = sp[pool.tuning]

        def v_sub_vote(rows):
            r = torch.as_tensor(rows, device=dev)
            return VoteAgg(dv.T[:, r], dv.C[:, r], pool.K)

        res["method"]["vote"] = selection.decider_block(
            pool, paths, VoteAgg(dv.T, dv.C, pool.K), dv.scorer,
            VoteAgg(dt.T, dt.C, pool.K), dt.scorer, dv.y, dt.y, dv.T, v_sub_vote, dom_v,
            info["q_free"])
        m = res["method"]["vote"]["main"]
        log(f"  method vote: {m['source']} k={m['k']} -> {m['test'][pool.metric]:.4f} (full "
            f"{res['method']['vote']['full_pool_test'][pool.metric]:.4f})  "
            f"[{time.time() - t0:.0f}s]")

    # probability averages
    res["sweep"]["avg"] = {}
    P_v = None
    if pool.tuning:
        P_v = pool.probs(pool.tuning, where="cuda")
        res["geometry"] = geometry.brier_geometry(P_v, sp[pool.tuning].y)
        curves, orders = avg_sweep(pool, sp[pool.tuning], P_v, R=10, seed=seed)
        res["sweep"]["avg"][pool.tuning] = {"curves": curves, "orders": orders}
        log(f"  geometry and avg sweep on {pool.tuning}  [{time.time() - t0:.0f}s]")
    on_gpu = pool.probs_bytes(pool.test) <= GPU_PROB_BUDGET
    P_t = pool.probs(pool.test, where="cuda" if on_gpu else "cpu")
    if not pool.tuning and on_gpu:                  # no tuning split: geometry on the test split
        res["geometry"] = geometry.brier_geometry(P_t, dt.y)
        res["geometry"]["split"] = pool.test
    P_g, rows = None, None
    if not on_gpu:
        rows = np.sort(np.random.default_rng(seed).choice(dt.n, GREEDY_SUBSAMPLE, replace=False))
        P_g = P_t[:, torch.as_tensor(rows)].to(dev)
    curves, orders = avg_sweep(pool, dt, P_t, R=10 if on_gpu else 5, seed=seed,
                               P_greedy=P_g, greedy_rows=rows)
    res["sweep"]["avg"][pool.test] = {"curves": curves, "orders": orders,
                                      "greedy_subsample": None if rows is None else len(rows)}
    del P_g
    torch.cuda.empty_cache()
    log(f"  avg sweep on {pool.test}  [{time.time() - t0:.0f}s]")
    if pool.tuning:
        def v_sub_avg(rows):
            return AvgAgg(P_v[:, torch.as_tensor(rows, device=dev)], dev)

        dv = sp[pool.tuning]
        dec = selection.decider_block(
            pool, paths, AvgAgg(P_v, dev), dv.scorer, AvgAgg(P_t, dev), dt.scorer,
            dv.y, dt.y, dv.T, v_sub_avg, dom_v, info["q_free"])
        res["method"]["avg"] = dec
        m = dec["main"]
        log(f"  method avg: {m['source']} k={m['k']} -> {m['test'][pool.metric]:.4f} (full "
            f"{dec['full_pool_test'][pool.metric]:.4f}); halves {dec['halves']}  "
            f"[{time.time() - t0:.0f}s]")
        noise_ind = weight_noise.load(pool, "inductive")
        if noise_ind is not None:
            res["inductive"] = selection.inductive_block(
                pool, dec["main"], AvgAgg(P_v, dev), dv.scorer, dv.y, dv.T, v_sub_avg, dom_v,
                dt.T, dt.C, P_t, dt.y, dev, noise_ind)
            log(f"  inductive check  [{time.time() - t0:.0f}s]")
        res["baselines"] = selection.baselines_block(
            pool, paths, dec, P_v, P_t, dv.T, dt.T, AvgAgg(P_v, dev), dv.scorer,
            AvgAgg(P_t, dev), dt.scorer, dv.y, dev)
        log(f"  baselines and budgets  [{time.time() - t0:.0f}s]")
    del P_v, P_t
    gc.collect()
    torch.cuda.empty_cache()
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="*")
    ap.add_argument("--all", action="store_true", help="every complete pool under REGADCOM_POOLS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args(argv)
    roots = list(a.pools)
    if a.all:
        roots = [os.path.join(POOLS, d) for d in sorted(os.listdir(POOLS))
                 if paper_pool(d)
                 and os.path.exists(os.path.join(POOLS, d, "predictions", "index.json"))]
    os.makedirs(OUT, exist_ok=True)
    failed = []
    for root in roots:
        pool = Pool(root)
        out = os.path.join(OUT, f"{pool.name}.json")
        if pool.config.get("mode", "build") != "build":
            log(f"{pool.name}: refined pool, left to the ancestry analysis")
            continue
        if a.skip_done and os.path.exists(out):
            log(f"{pool.name}: already done")
            continue
        if pool.M < pool.config.get("pool_size", pool.M):
            log(f"{pool.name}: {pool.M}/{pool.config['pool_size']} members, skipped until complete")
            continue
        log(f"\n=== {pool.name}: {pool.benchmark}, K={pool.K}, M={pool.M}, metric {pool.metric}")
        try:
            res = analyse(pool, seed=a.seed)
        except Exception:
            traceback.print_exc()
            failed.append(pool.name)
            continue
        with open(out + ".tmp", "w", encoding="utf-8") as f:
            json.dump(clean(res), f)
        os.replace(out + ".tmp", out)
        log(f"  wrote {out} ({res['seconds'] / 60:.1f} min)")
        gc.collect()
        torch.cuda.empty_cache()
    if failed:
        log(f"\nfailed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
