"""Collision statistics, regime masses and the Brier geometry of a pool on one split.

Collision probabilities are pooled over member pairs and points,

    q     = sum_x sum_l C(c_l(x), 2) / sum_x C(w(x), 2)
    q^(3) = sum_x sum_l C(c_l(x), 3) / sum_x C(w(x), 3)

where w(x) is the number of members that disagree with the reference label at x
and c_l(x) how many of them chose label l.  With the true label as reference this
is the collision probability of the paper; with the pool plurality z(x) as
reference it is the label-free estimate of Proposition 20.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _labels(ref, dev):
    if torch.is_tensor(ref):
        return ref.to(dev).long()
    return torch.as_tensor(np.asarray(ref), dtype=torch.long, device=dev)


def disagreement_counts(top1, ref, K):
    """(N, K) int32: how many members chose each label at each point, among those != ref."""
    M, N = top1.shape
    ref = _labels(ref, top1.device)
    counts = torch.zeros((N, K), dtype=torch.int32, device=top1.device)
    ar = torch.arange(N, device=top1.device)
    for i in range(M):
        t = top1[i]
        m = t != ref
        counts.index_put_((ar[m], t[m]), torch.ones(int(m.sum()), dtype=torch.int32,
                                                    device=top1.device), accumulate=True)
    return counts


def collision_stats(top1, ref, K):
    """Pooled pairwise and three-fold collision, per-point spread, and regime masses."""
    M, N = top1.shape
    c = disagreement_counts(top1, ref, K).double()
    w = c.sum(dim=1)
    pairs_same = (c * (c - 1) / 2).sum()
    pairs_both = (w * (w - 1) / 2).sum()
    trip_same = (c * (c - 1) * (c - 2) / 6).sum()
    trip_both = (w * (w - 1) * (w - 2) / 6).sum()
    q = float(pairs_same / pairs_both) if pairs_both > 0 else float("nan")
    q3 = float(trip_same / trip_both) if trip_both > 0 else float("nan")
    p = 1.0 - w / M                                            # fraction agreeing with ref
    pi_max = torch.where(w > 0, c.max(dim=1).values / w.clamp_min(1), torch.zeros_like(w))
    mu = p - (1.0 - p) * pi_max                                # expected vote margin
    n_pairs = M * (M - 1) / 2
    return {
        "q": q,
        "q3": q3,
        "both_wrong_rate": float(pairs_both / (n_pairs * N)),
        "mean_member_error": float((w / M).mean()),
        "frac_majority_wrong": float((p < 0.5).double().mean()),
        "frac_mu_negative": float((mu < 0).double().mean()),
        "mass_p_mu_negative": float((p * (mu < 0).double()).mean()),
        "frac_all_wrong": float((w == M).double().mean()),
        "frac_all_right": float((w == 0).double().mean()),
        "p_hist": torch.bincount((M - w).long(), minlength=M + 1).cpu().numpy().tolist(),
    }


def label_free_bounds(top1, z, y, K, q_labels, q_free, eta=0.05):
    """The terms of Proposition 20 and the sampling bound (18), pooled over pairs."""
    M, N = top1.shape
    z_t = _labels(z, top1.device)
    y_t = _labels(y, top1.device)
    e_z = float((z_t != y_t).double().mean())
    w = (top1 != z_t[None, :]).sum(dim=0).double()
    beta = float((w * (w - 1) / 2).sum() / (M * (M - 1) / 2 * N))
    t = math.sqrt(math.log(2 * M * M / eta) / (2 * N))
    return {"e_z": e_z, "beta": beta, "q_labels": q_labels, "q_free": q_free,
            "abs_error": abs(q_free - q_labels),
            "bias_bound": 2 * e_z / beta if beta > 0 else float("inf"),
            "sampling_bound": 2 * t / beta if beta > 0 else float("inf")}


@torch.no_grad()
def brier_geometry(P, y, chunk_bytes=256 * 2 ** 20):
    """Ambiguity decomposition of the full-pool average and error alignment.

    P is (M, n, K) on the GPU.  With e_i(x) = h_i(x) - e_y, the Gram matrix of the
    error vectors is G_ij = sum_x <h_i, h_j> - h_i[y] - h_j[y] + 1.
    """
    M, n, K = P.shape
    dev = P.device
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=dev)
    step = max(1, int(chunk_bytes // (4 * M * K)))
    G = torch.zeros((M, M), dtype=torch.float64, device=dev)
    Yg = torch.zeros((M, M), dtype=torch.float64, device=dev)
    a = torch.zeros(M, dtype=torch.float64, device=dev)
    for s in range(0, n, step):
        Pc = P[:, s:s + step].float()
        flat = Pc.reshape(M, -1)
        G += (flat @ flat.T).double()
        idx = torch.arange(Pc.shape[1], device=dev)
        py = Pc[:, idx, yt[s:s + step]]
        a += py.sum(dim=1).double()
        Yg += (py @ py.T).double()
    E = G - a[:, None] - a[None, :] + n
    diag = torch.diagonal(E)
    mean_risk = float(diag.mean() / n)
    committee_risk = float(E.sum() / (M * M * n))
    cos = E / torch.sqrt(diag[:, None] * diag[None, :])
    off = ~torch.eye(M, dtype=torch.bool, device=dev)
    # error vectors restricted to the wrong-label coordinates (h_i(x) without its true
    # label's entry), so that member accuracy does not enter through e_y
    Ew = G - Yg
    dw = torch.diagonal(Ew)
    cos_w = Ew / torch.sqrt(dw[:, None] * dw[None, :]).clamp_min(1e-12)
    return {"mean_member_brier": mean_risk, "committee_brier": committee_risk,
            "ambiguity": mean_risk - committee_risk,
            "ambiguity_share": (mean_risk - committee_risk) / mean_risk,
            "error_alignment": float(cos[off].mean()),
            "error_alignment_wrong": float(cos_w[off].mean())}


def calibration(top1, conf, y, bins=15):
    """Expected calibration error of each member's top-1 confidence (equal-width bins)."""
    yt = _labels(y, top1.device)
    out = []
    edges = torch.linspace(0, 1, bins + 1, device=top1.device)
    for i in range(top1.shape[0]):
        c = conf[i].float()
        right = (top1[i] == yt).double()
        b = torch.bucketize(c, edges[1:-1])
        acc_b = torch.bincount(b, weights=right, minlength=bins)
        conf_b = torch.bincount(b, weights=c.double(), minlength=bins)
        out.append(float((acc_b - conf_b).abs().sum() / len(c)))
    out = np.asarray(out)
    return {"ece_mean": float(out.mean()), "ece_min": float(out.min()),
            "ece_max": float(out.max())}


def split_sample_mu(top1, ref, K, seed=0):
    """Share of points with a negative vote margin, without the max-over-few-votes bias.

    The members are split at random; the strongest wrong label is chosen on one half
    and its frequency, with the share of correct members, is measured on the other;
    both directions are averaged."""
    M, N = top1.shape
    perm = np.random.default_rng(seed).permutation(M)
    halves = (perm[:M // 2], perm[M // 2:])
    ref_t = _labels(ref, top1.device)
    out = []
    for a, b in (halves, halves[::-1]):
        ca = disagreement_counts(top1[torch.as_tensor(a, device=top1.device)], ref, K)
        top_wrong = ca.argmax(dim=1)
        has = ca.max(dim=1).values > 0
        Tb = top1[torch.as_tensor(b, device=top1.device)]
        p = (Tb == ref_t[None]).float().mean(dim=0)
        on_top = (Tb == top_wrong[None]).float().sum(dim=0)
        wrong = (Tb != ref_t[None]).float().sum(dim=0)
        pi = torch.where(has & (wrong > 0), on_top / wrong.clamp_min(1), torch.zeros_like(p))
        mu = p - (1 - p) * pi
        out.append(float((mu < 0).float().mean()))
    return float(np.mean(out))


def pair_decomposition(top1, z, y, K):
    """Pooled pair masses of Proposition 20: A, B, G, c_G, R, S (pairs x points)."""
    M, N = top1.shape
    z_t, y_t = _labels(z, top1.device), _labels(y, top1.device)
    counts = torch.zeros((N, K), dtype=torch.float64, device=top1.device)
    ar = torch.arange(N, device=top1.device)
    for i in range(M):
        counts[ar, top1[i]] += 1.0
    n_y = counts[ar, y_t]
    n_z = counts[ar, z_t]
    zw = (z_t != y_t).double()
    W1 = (M - n_y) - n_z * zw                         # wrong members that also leave z
    c2 = lambda v: v * (v - 1) / 2
    others = counts.clone()
    others[ar, y_t] = 0
    others[ar, z_t] = 0
    out = {"A": c2(n_z) * zw, "B": n_z * W1 * zw, "G": c2(W1),
           "cG": c2(others).sum(dim=1), "R": c2(n_y) * zw, "S": n_y * W1 * zw}
    return {k: float(v.sum()) for k, v in out.items()}


def corrected_q(dec_t, dec_v):
    """q-hat on the target, with the excluded masses estimated as ratios to G on V."""
    rA, rB, rR, rS = (dec_v[k] / dec_v["G"] for k in ("A", "B", "R", "S"))
    N_D = dec_t["G"] + dec_t["R"] + dec_t["S"]          # observable without labels
    N_agree = dec_t["cG"] + dec_t["R"]                  # observable without labels
    G = N_D / (1 + rR + rS)
    cG = N_agree - rR * G
    return (cG + rA * G) / (G + rA * G + rB * G)
