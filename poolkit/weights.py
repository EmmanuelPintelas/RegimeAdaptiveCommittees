"""Data-free statistics of a member's weights, recorded when it is committed.

The spectral statistics are proposers in the paper, so they are computed once
from the kept checkpoint and stored with the member rather than recomputed from
a hundred checkpoints later.  They are comparable only within one architecture.

    w_std, w_l2   spread and norm of all convolution and linear weights, pooled
    head_norm     Frobenius norm of the classifier head
    spec_norm     mean top singular value over layers
    stable_rank   mean ||W||_F^2 / ||W||_2^2 over layers
    alpha         mean Hill estimate of each layer's eigenvalue tail exponent
"""
from __future__ import annotations

import numpy as np
import torch


def hill_alpha(eig, frac=0.2):
    eig = np.sort(eig[eig > 0])[::-1]
    k = max(int(len(eig) * frac), 3)
    if len(eig) <= k:
        return float("nan")
    return float(1.0 + k / np.sum(np.log(eig[:k] / eig[k])))


@torch.no_grad()
def weight_stats(state_dict):
    pooled, specs, sranks, alphas = [], [], [], []
    head_norm = float("nan")
    for key, t in state_dict.items():
        if not t.is_floating_point() or t.dim() < 2:
            continue
        w = t.detach().to("cpu", torch.float64)
        if key.startswith("head.") and key.endswith("weight"):
            head_norm = float(w.norm())
        pooled.append(w.flatten())
        m = w.reshape(w.shape[0], -1)
        if min(m.shape) < 2:
            continue
        s = torch.linalg.svdvals(m)
        s2 = (s ** 2).numpy()
        specs.append(float(s[0]))
        if s2[0] > 0:
            sranks.append(float(s2.sum() / s2[0]))
        alphas.append(hill_alpha(s2 / m.shape[1]))
    allw = torch.cat(pooled)
    return {"w_std": float(allw.std()), "w_l2": float(allw.norm()), "head_norm": head_norm,
            "spec_norm": float(np.nanmean(specs)), "stable_rank": float(np.nanmean(sranks)),
            "alpha": float(np.nanmean(alphas)), "n_layers": len(specs)}
