"""Selection with a tuning split (Algorithm 1) and its baselines, on stored predictions.

decider_block    the decider of Algorithm 1 under one aggregation: held-out search
                 families, regime and uniform priors, refit and conservative
                 deployments, the menu and family ablations, budgets k <= k_max,
                 and the transfer diagnostic of Proposition 19 against the full pool.
baselines_block  the standard tuning-split baselines (best member, top-k, Caruana
                 ensemble selection, log-odds weighted vote, linear-pool stacking),
                 selection by proper scores, and the budgeted comparison; probability
                 averaging only.

Every committee is chosen on the tuning split V (or without labels) and scored once
on the test split.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import method, proposers
from .committees import Scorer, greedy, member_scores, path_scores

BUDGETS = (5, 10, 20)
CARUANA = {"init": 5, "steps": 100, "bags": 20, "bag_frac": 0.5, "seed": 0}
RANDOM_DRAWS = 20


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def score_members(agg, members, scorer, metrics):
    """Scores of one committee (the last prefix of its own path)."""
    c = path_scores(agg, list(members), scorer, metrics)
    return {m: c[m][-1] for m in metrics}


def weighted_avg_pred(P, w, dev):
    """argmax_c sum_i w_i P_i(x)_c, members streamed to the device one at a time."""
    S = None
    for i in np.flatnonzero(np.asarray(w) > 0):
        part = P[int(i)].to(dev, non_blocking=True).float() * float(w[i])
        S = part if S is None else S + part
    return S.argmax(dim=1)


def weighted_vote_pred(T, w, K):
    """Plurality with member weights w over the top-1 labels T (M, N)."""
    N = T.shape[1]
    ar = torch.arange(N, device=T.device)
    score = torch.zeros((N, K), dtype=torch.float64, device=T.device)
    for i in np.flatnonzero(np.asarray(w) > 0):
        score.index_put_((ar, T[int(i)]), torch.full((N,), float(w[i]), dtype=torch.float64,
                                                      device=T.device), accumulate=True)
    return score.argmax(dim=1)


def held_out_families(M, K, T_v, v_sub, y_v, dom_v, dev):
    """Both search families, each built on one half of V and scored on the other.

    Returns ({family: {"score", "halves", "orders", "curves"}}, note)."""
    A, B, note = method.domain_halves(dom_v)
    out = {f: {"score": np.zeros(M), "halves": [], "orders": [], "curves": []}
           for f in method.FAMILIES}
    for fit, sc in ((A, B), (B, A)):
        fit_scorer = Scorer(y_v[fit], K, dev)
        agg = v_sub(fit)
        orders = {"held-out greedy": greedy(agg, fit_scorer, "balanced_acc")[0]}
        del agg
        ms = member_scores(T_v[:, torch.as_tensor(fit, device=T_v.device)], fit_scorer,
                           "balanced_acc")
        orders["held-out top-k"] = [int(j) for j in np.argsort(-ms, kind="stable")]
        agg = v_sub(sc)
        sc_scorer = Scorer(y_v[sc], K, dev)
        w = len(sc) / len(y_v)
        counts = method.class_counts(y_v[sc])
        for f, o in orders.items():
            curve = path_scores(agg, o, sc_scorer, ("balanced_acc",))["balanced_acc"]
            out[f]["score"] += w * np.asarray(curve)
            out[f]["halves"].append((w, counts))
            out[f]["orders"].append(o)
            out[f]["curves"].append(curve)
        del agg
        torch.cuda.empty_cache()
    return out, note


def transfer_terms(v_agg, t_agg, y_v, y_t, comm, M):
    """Proposition 19's quantities for a committee against the full pool, plain error."""
    def preds(agg, members):
        agg.reset()
        for j in members:
            agg.add(j)
        return agg.pred.clone()

    out = {"reference": "full pool"}
    for tag, agg, y in (("V", v_agg, y_v), ("T", t_agg, y_t)):
        y = torch.as_tensor(y, device=agg.pred.device)
        c, f = preds(agg, comm), preds(agg, range(M))
        dis = c != f
        d = float(dis.float().mean())
        out[f"dis_{tag}"] = d
        out[f"a_{tag}"] = (float(((c != y) & dis).float().mean() - ((f != y) & dis).float().mean()) / d
                           if d > 0 else None)
    if out["a_V"] is None or out["a_T"] is None:
        out.update(gamma=None, regret=0.0)
    else:
        out["gamma"] = abs(out["a_T"] - out["a_V"])
        out["regret"] = out["a_T"] * out["dis_T"]
    return out


# --------------------------------------------------------------------------- #
# the decider
# --------------------------------------------------------------------------- #
def decider_block(pool, paths, v_agg, v_scorer, t_agg, t_scorer, y_v, y_t, T_v, v_sub,
                  dom_v, q_free):
    """Algorithm 1 under one aggregation, its ablations and budgets, scored on the test split."""
    M, dev = pool.M, v_scorer.y.device
    metrics = (pool.metric, "balanced_acc", "acc")
    v_curves = {n: path_scores(v_agg, o, v_scorer, metrics) for n, o in paths.items()}
    t_curves = {n: path_scores(t_agg, o, t_scorer, metrics) for n, o in paths.items()}
    held, note = held_out_families(M, pool.K, T_v, v_sub, y_v, dom_v, dev)
    counts_v = method.class_counts(y_v)
    refit = {"held-out greedy": greedy(v_agg, v_scorer, "balanced_acc")[0],
             "held-out top-k": [int(j) for j in np.argsort(
                 -member_scores(T_v, v_scorer, "balanced_acc"), kind="stable")]}
    refit_t = {f: path_scores(t_agg, o, t_scorer, metrics) for f, o in refit.items()}
    refit_v = {f: path_scores(v_agg, o, v_scorer, ("balanced_acc",))["balanced_acc"]
               for f, o in refit.items()}
    full_t = {m: t_curves[next(iter(paths))][m][-1] for m in metrics}

    def test_of(source, k):
        if source in method.FAMILIES:
            return {m: refit_t[source][m][k - 1] for m in metrics}
        if source == "full pool":
            return dict(full_t)
        return {m: t_curves[source][m][k - 1] for m in metrics}

    def members_of(source, k):
        if source in method.FAMILIES:
            return refit[source][:k]
        if source == "full pool":
            return list(range(M))
        return paths[source][:k]

    def conservative(dec):
        src, k = dec["source"], dec["k"]
        if src not in method.FAMILIES:
            return None
        h = held[src]
        idx = int(np.argmax([c[k - 1] for c in h["curves"]]))
        comm = h["orders"][idx][:k]
        return {"members": comm, "test": score_members(t_agg, comm, t_scorer, metrics)}

    def record(dec):
        d = dict(dec)
        d["test"] = test_of(dec["source"], dec["k"])
        d["members"] = members_of(dec["source"], dec["k"])
        return d

    menu = proposers.menu(paths)
    prop = {n: v_curves[n]["balanced_acc"] for n in menu}
    main = record(method.decide(prop, held, counts_v, M, q=q_free))
    main["conservative"] = conservative(main)
    main["transfer"] = transfer_terms(v_agg, t_agg, y_v, y_t, main["members"], M)
    uniform = record(method.decide(prop, held, counts_v, M, q=None))
    uniform["conservative"] = conservative(uniform)

    ablations = {}
    for mname in proposers.MENUS:
        ps = {n: v_curves[n]["balanced_acc"] for n in proposers.menu(paths, mname)}
        ablations[mname] = record(method.decide(ps, held, counts_v, M, q=q_free))
    for mname in proposers.NO_SUBSET_MENUS:
        ps = {n: v_curves[n]["balanced_acc"] for n in proposers.menu(paths, mname)}
        ablations[f"{mname} no search"] = record(method.decide(ps, None, counts_v, M, q=q_free))
    for fam in method.FAMILIES:
        ablations[f"menu, {fam} only"] = record(
            method.decide(prop, {fam: held[fam]}, counts_v, M, q=q_free))
    full_v = v_curves[next(iter(paths))]["balanced_acc"][-1]
    two = record(method.decide(None, {"held-out greedy": held["held-out greedy"]}, counts_v, M,
                               q=q_free, fixed={"full pool": full_v}))
    ablations["full pool vs held-out greedy"] = two

    budgets = {}
    for b in BUDGETS:
        budgets[b] = record(method.decide(prop, held, counts_v, M, q=q_free, kmax=b))

    # greedy subset search scored in-sample on V, at its best size there: the rule the
    # held-out test replaces, with its transfer terms
    kg = int(np.argmax(refit_v["held-out greedy"])) + 1
    insample = {"source": "greedy on V", "k": kg, "members": refit["held-out greedy"][:kg],
                "test": test_of("held-out greedy", kg)}
    insample["transfer"] = transfer_terms(v_agg, t_agg, y_v, y_t, insample["members"], M)

    return {
        "main": main, "uniform_prior": uniform, "ablations": ablations, "budgets": budgets,
        "insample_greedy": insample,
        "halves": note, "q_free": q_free,
        "held_out": {f: {"score": h["score"], "curves": h["curves"]} for f, h in held.items()},
        "refit_v": refit_v, "refit_orders": refit,
        "full_pool_test": full_t,
        "per_proposer": method.per_proposer(v_curves, t_curves, pool.metric),
        "v_curves": {n: c["balanced_acc"] for n, c in v_curves.items()},
        "t_curves": {n: c[pool.metric] for n, c in t_curves.items()},
        "t_curves_balanced": {n: c["balanced_acc"] for n, c in t_curves.items()},
    }


# --------------------------------------------------------------------------- #
# baselines (probability averaging)
# --------------------------------------------------------------------------- #
def caruana(v_agg, v_scorer, ms_v, candidates, init, steps, cap=None):
    """Forward selection with replacement from a sorted initialisation (Caruana et al. 2004).

    Returns member counts (M,) of the prefix with the best balanced accuracy on V.
    ``cap`` bounds the number of distinct members."""
    M = len(ms_v)
    cand = list(candidates)
    start = sorted(cand, key=lambda j: (-ms_v[j], j))[:min(init, cap or init)]
    v_agg.reset()
    for j in start:
        v_agg.add(j)
    chosen = list(start)
    best_s, best_len = v_scorer(v_agg.pred, "balanced_acc"), len(chosen)
    for _ in range(steps - len(start)):
        pool_j = cand if cap is None or len(set(chosen)) < cap else sorted(set(chosen))
        s_best, j_best = -np.inf, None
        for j in pool_j:
            s = v_scorer(v_agg.trial(j), "balanced_acc")
            if s > s_best:
                s_best, j_best = s, j
        v_agg.add(j_best)
        chosen.append(j_best)
        if s_best > best_s + 1e-12:
            best_s, best_len = s_best, len(chosen)
    counts = np.zeros(M)
    for j in chosen[:best_len]:
        counts[j] += 1
    return counts


def linear_pool_weights(htrue, iters=500):
    """Mixture weights w >= 0, sum 1, maximising the likelihood sum_j log sum_i w_i h_i(x_j)_{y_j} (EM)."""
    H = htrue.double().clamp_min(1e-12)                     # (M, n)
    w = torch.full((H.shape[0],), 1.0 / H.shape[0], dtype=torch.float64, device=H.device)
    for _ in range(iters):
        mix = (w[:, None] * H).sum(dim=0)
        w = (w[:, None] * H / mix[None, :]).mean(dim=1)
    return w.cpu().numpy()


def proper_curves(P, y, order, dev):
    """NLL and Brier score of the averaged committee along a path."""
    y_t = torch.as_tensor(np.asarray(y), device=dev)
    ar = torch.arange(len(y_t), device=dev)
    S = None
    nll, brier = [], []
    for k, j in enumerate(order, start=1):
        p = P[int(j)].to(dev, non_blocking=True).float()
        S = p if S is None else S + p
        avg = S / k
        py = avg[ar, y_t]
        nll.append(float(-torch.log(py.clamp_min(1e-12)).mean()))
        brier.append(float(((avg ** 2).sum(dim=1) - 2 * py + 1).mean()))
    return nll, brier


def baselines_block(pool, paths, dec, P_v, P_t, T_v, T_t, v_agg, v_scorer, t_agg, t_scorer,
                    y_v, dev):
    """Tuning-split baselines, proper-score selection and budgets, scored on the test split."""
    M, K = pool.M, pool.K
    metrics = (pool.metric, "balanced_acc", "acc")
    ms_v = member_scores(T_v, v_scorer, "balanced_acc")
    out = {}

    best = int(np.argmax(ms_v))
    out["best member on V"] = {"k": 1, "members": [best],
                              "test": score_members(t_agg, [best], t_scorer, metrics)}
    topk = dec["refit_orders"]["held-out top-k"]
    kv = int(np.argmax(dec["refit_v"]["held-out top-k"])) + 1
    out["top-k on V"] = {"k": kv, "members": topk[:kv],
                         "test": score_members(t_agg, topk[:kv], t_scorer, metrics)}
    gv = dec["refit_orders"]["held-out greedy"]
    kg = int(np.argmax(dec["refit_v"]["held-out greedy"])) + 1
    out["greedy on V"] = {"k": kg, "members": gv[:kg],
                          "test": score_members(t_agg, gv[:kg], t_scorer, metrics)}

    rng = np.random.default_rng(CARUANA["seed"])
    counts = np.zeros(M)
    for _ in range(CARUANA["bags"]):
        bag = sorted(rng.choice(M, int(M * CARUANA["bag_frac"]), replace=False).tolist())
        counts += caruana(v_agg, v_scorer, ms_v, bag, CARUANA["init"], CARUANA["steps"])
    pred = weighted_avg_pred(P_t, counts, dev)
    out["Caruana (bagged)"] = {"k": int((counts > 0).sum()), "weights": counts,
                               "test": t_scorer.all(pred)}

    acc_v = np.clip(member_scores(T_v, v_scorer, "acc"), 1e-3, 1 - 1e-3)
    w_lo = np.maximum(0.0, np.log((K - 1) * acc_v / (1 - acc_v)))
    pred = weighted_vote_pred(T_t, w_lo, K)
    out["log-odds weighted vote"] = {"k": int((w_lo > 0).sum()), "weights": w_lo, "rule": "vote",
                                     "test": t_scorer.all(pred)}

    y_vt = torch.as_tensor(np.asarray(y_v), device=dev)
    ar = torch.arange(len(y_vt), device=dev)
    htrue = torch.stack([P_v[i].to(dev).float()[ar, y_vt] for i in range(M)])
    w_lp = linear_pool_weights(htrue)
    pred = weighted_avg_pred(P_t, np.where(w_lp > 1e-6, w_lp, 0.0), dev)
    out["linear-pool stacking"] = {"k": int((w_lp > 1e-6).sum()),
                                   "weights": np.where(w_lp > 1e-6, w_lp, 0.0),
                                   "test": t_scorer.all(pred)}
    del htrue

    menu = proposers.menu(paths)
    proper = {}
    for n in menu:
        nll, brier = proper_curves(P_v, y_v, paths[n], dev)
        proper[n] = {"nll": nll, "brier": brier}
    for crit in ("nll", "brier"):
        name, k = min(((n, int(np.argmin(c[crit])) + 1) for n, c in proper.items()),
                      key=lambda t: proper[t[0]][crit][t[1] - 1])
        out[f"proposed paths by {crit.upper()}"] = {
            "source": name, "k": k,
            "test": {m: (dec["t_curves"][name][k - 1] if m == pool.metric else
                         dec["t_curves_balanced"][name][k - 1] if m == "balanced_acc" else None)
                     for m in metrics}}
    name, k = max(((n, int(np.argmax(dec["v_curves"][n])) + 1) for n in menu),
                  key=lambda t: dec["v_curves"][t[0]][t[1] - 1])
    out["proposed paths by balanced accuracy"] = {
        "source": name, "k": k,
        "test": {pool.metric: dec["t_curves"][name][k - 1],
                 "balanced_acc": dec["t_curves_balanced"][name][k - 1]}}

    budget = {}
    rng = np.random.default_rng(1)
    for b in BUDGETS:
        row = {"decider": dec["budgets"][b]["test"]}
        draws = [score_members(t_agg, rng.choice(M, b, replace=False).tolist(), t_scorer, metrics)
                 for _ in range(RANDOM_DRAWS)]
        row["random"] = {m: float(np.mean([d[m] for d in draws])) for m in metrics}
        row["source-validation top-k"] = score_members(t_agg, paths["P6 source val"][:b], t_scorer,
                                                       metrics)
        row["top-k on V"] = score_members(t_agg, topk[:b], t_scorer, metrics)
        row["greedy on V"] = score_members(t_agg, gv[:b], t_scorer, metrics)
        cnt = caruana(v_agg, v_scorer, ms_v, range(M), CARUANA["init"], CARUANA["steps"], cap=b)
        pred = weighted_avg_pred(P_t, cnt, dev)
        row["Caruana (capped)"] = t_scorer.all(pred)
        row["Caruana (capped) weights"] = cnt
        row["confidence top-k"] = score_members(t_agg, paths["P1 confidence"][:b], t_scorer, metrics)
        for n in menu:
            row[n] = score_members(t_agg, paths[n][:b], t_scorer, metrics)
        budget[b] = row
    out["budget"] = budget
    out["proper_curves"] = proper
    return out


def decide_main(pool, paths, v_agg, v_scorer, y_v, T_v, v_sub, dom_v, q):
    """The decider's choice alone (no test evaluation): {"source", "k", ..., "members"}."""
    M, dev = pool.M, v_scorer.y.device
    held, _ = held_out_families(M, pool.K, T_v, v_sub, y_v, dom_v, dev)
    prop = {n: path_scores(v_agg, o, v_scorer, ("balanced_acc",))["balanced_acc"]
            for n, o in proposers.menu(paths).items()}
    dec = method.decide(prop, held, method.class_counts(y_v), M, q=q)
    if dec["source"] == "held-out greedy":
        dec["members"] = greedy(v_agg, v_scorer, "balanced_acc")[0][:dec["k"]]
    elif dec["source"] == "held-out top-k":
        order = np.argsort(-member_scores(T_v, v_scorer, "balanced_acc"), kind="stable")
        dec["members"] = [int(j) for j in order[:dec["k"]]]
    else:
        dec["members"] = paths[dec["source"]][:dec["k"]]
    return dec


def inductive_block(pool, dec_main, v_agg, v_scorer, y_v, T_v, v_sub, dom_v, T_t, C_t, P_t,
                    y_t, dev, noise):
    """The proposers read one half of the test inputs (whole domains); the decider's
    committee is scored on the other half, next to the transductive committee and the full
    pool on the same half."""
    from .committees import RowAvgAgg
    A, B, note = method.domain_halves(pool.domains(pool.test), seed=0)
    a = torch.as_tensor(A, device=T_t.device)
    paths_i, info_i = proposers.proposer_paths(pool, T_t[:, a], C_t[:, a], noise=noise)
    dec = decide_main(pool, paths_i, v_agg, v_scorer, y_v, T_v, v_sub, dom_v, info_i["q_free"])
    agg = RowAvgAgg(P_t, B, dev)
    sc = Scorer(y_t[B], pool.K, dev)
    metrics = (pool.metric, "balanced_acc", "acc")
    return {"halves": note, "q_free": info_i["q_free"],
            "inductive": {"source": dec["source"], "k": dec["k"], "members": dec["members"],
                          "test": score_members(agg, dec["members"], sc, metrics)},
            "transductive": {"source": dec_main["source"], "k": dec_main["k"],
                             "test": score_members(agg, dec_main["members"], sc, metrics)},
            "full_pool": score_members(agg, range(pool.M), sc, metrics)}
