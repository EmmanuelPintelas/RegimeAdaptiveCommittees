"""Scores for a member or a committee on one split, in numpy.

Computing these decision metrics with scikit-learn takes tens of seconds per call on
a 1139-class split; a pool of a hundred members scored on every split after every
commit cannot afford that.  Recall, precision and F1 are averaged over the classes
present in the labels, the geometric mean clips recall at 1e-6, and ECE uses fifteen
equal-width bins.  NLL, mean confidence and the share of the most frequent predicted class
are added for the gate and the report.
"""
from __future__ import annotations

import numpy as np


def log_softmax(logits):
    z = np.asarray(logits, dtype=np.float32)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def decision_metrics(pred, labels, K):
    pred = np.asarray(pred, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    present = np.unique(labels)
    tc = np.bincount(labels, minlength=K)[present].astype(np.float64)
    pc = np.bincount(pred, minlength=K)[present].astype(np.float64)
    tp = np.bincount(labels[pred == labels], minlength=K)[present].astype(np.float64)
    rec = tp / tc
    prec = np.divide(tp, pc, out=np.zeros_like(tp), where=pc > 0)
    s = prec + rec
    f1 = np.divide(2.0 * prec * rec, s, out=np.zeros_like(s), where=s > 0)
    n = len(labels)
    return {
        "acc": float(np.mean(pred == labels)),
        "balanced_acc": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "gm": float(np.exp(np.mean(np.log(np.clip(rec, 1e-6, None))))),
        "pred_share_max": float(np.bincount(pred, minlength=K).max() / n),
        "n": int(n),
        "k_present": int(len(present)),
    }


def ece(conf, correct, n_bins=15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def score_probs(p, labels, K, logp=None):
    labels = np.asarray(labels, dtype=np.int64)
    pred = p.argmax(axis=1)
    out = decision_metrics(pred, labels, K)
    idx = np.arange(len(labels))
    py = p[idx, labels].astype(np.float64)
    sq = np.einsum("ij,ij->i", p, p).astype(np.float64)
    out["brier"] = float(np.mean(sq - 2.0 * py + 1.0))
    lp = (np.log(np.clip(py, 1e-12, None)) if logp is None
          else logp[idx, labels].astype(np.float64))
    out["nll"] = float(-lp.mean())
    conf = p.max(axis=1).astype(np.float64)
    out["confidence"] = float(conf.mean())
    out["ece"] = ece(conf, (pred == labels).astype(np.float64))
    return out


def score_logits(logits, labels, K):
    logp = log_softmax(logits)
    return score_probs(np.exp(logp), labels, K, logp=logp)


def pairwise_stats(top1, labels):
    """Mean pairwise disagreement and the pooled collision probability.

    The collision probability is P(two members predict the same label | both
    are wrong at that point), pooled over member pairs and points.
    """
    P = np.asarray(top1)
    labels = np.asarray(labels)
    M, N = P.shape
    if M < 2:
        return {"mean_pairwise_disagreement": None, "both_wrong_rate": None,
                "collision_probability": None}
    W = P != labels[None, :]
    pairs = M * (M - 1) // 2
    disagree = both = same = 0
    for i in range(M - 1):
        eq = P[i + 1:] == P[i]
        bw = W[i + 1:] & W[i]
        disagree += int((~eq).sum())
        both += int(bw.sum())
        same += int((eq & bw).sum())
    return {"mean_pairwise_disagreement": disagree / (pairs * N),
            "both_wrong_rate": both / (pairs * N),
            "collision_probability": (same / both) if both else None}


def spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else None
