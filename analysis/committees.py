"""Committees: aggregation rules, incremental construction, greedy search, scoring.

Three aggregations, as in the paper:

    vote   plurality of top-1 labels; ties between labels with equally many votes
           are broken by the summed top-1 probability of their voters
    wvote  confidence-weighted vote: each member votes with its top-1 probability
    avg    average of the probability vectors

The zero-one error the theory is stated for (plurality, ties counted as errors)
is tracked separately by PluralityCounter.

Committees are grown one member at a time.  For the two vote rules adding a
member raises one entry per point, so the effect of any candidate on the
committee's predictions is computed in O(N) without touching the other labels;
that is what makes greedy search over a hundred members and a thousand classes
cheap.  The average needs the full probabilities and runs on the GPU.
"""
from __future__ import annotations

import numpy as np
import torch

METRICS = ("acc", "balanced_acc", "macro_f1", "gm")


class Scorer:
    """The decision metrics of poolkit.metrics, on the GPU, for one split.

    Recall, precision and F1 are averaged over the classes present in the
    labels; the geometric mean clips recall at 1e-6.
    """

    def __init__(self, y, K, dev):
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=dev)
        self.K = int(K)
        self.n = int(self.y.numel())
        self.present = torch.unique(self.y)
        self.tc = torch.bincount(self.y, minlength=self.K)[self.present].double()

    def _tp(self, pred):
        return torch.bincount(self.y[pred == self.y], minlength=self.K)[self.present].double()

    def __call__(self, pred, metric):
        if metric == "acc":
            return float((pred == self.y).double().mean())
        tp = self._tp(pred)
        rec = tp / self.tc
        if metric == "balanced_acc":
            return float(rec.mean())
        if metric == "gm":
            return float(torch.exp(torch.log(rec.clamp_min(1e-6)).mean()))
        if metric == "macro_f1":
            pc = torch.bincount(pred, minlength=self.K)[self.present].double()
            prec = torch.where(pc > 0, tp / pc.clamp_min(1.0), torch.zeros_like(tp))
            s = prec + rec
            f1 = torch.where(s > 0, 2.0 * prec * rec / s.clamp_min(1e-12), torch.zeros_like(s))
            return float(f1.mean())
        raise ValueError(f"unknown metric {metric!r}")

    def all(self, pred):
        return {m: self(pred, m) for m in METRICS}

    def recalls(self, pred):
        return (self._tp(pred) / self.tc).cpu().numpy()


class VoteAgg:
    """Plurality ('vote') or confidence-weighted ('wvote') committee, grown incrementally."""

    BIG = 1000.0          # larger than any possible sum of confidences over 100 members

    def __init__(self, top1, conf, K, weighted=False):
        self.T = top1
        self.C = conf
        self.M, self.N = top1.shape
        self.K = int(K)
        self.weighted = weighted
        self.ar = torch.arange(self.N, device=top1.device)
        self.reset()

    def reset(self):
        self.score = torch.zeros((self.N, self.K), dtype=torch.float32, device=self.T.device)
        self.mx = torch.full((self.N,), -1.0, dtype=torch.float32, device=self.T.device)
        self.pred = torch.zeros(self.N, dtype=torch.long, device=self.T.device)
        self.members = []

    def _w(self, j):
        return self.C[j] if self.weighted else self.C[j] + self.BIG

    def trial(self, j):
        val = self.score[self.ar, self.T[j]] + self._w(j)
        return torch.where(val > self.mx, self.T[j], self.pred)

    def add(self, j):
        t = self.T[j]
        self.score.index_put_((self.ar, t), self._w(j), accumulate=True)
        val = self.score[self.ar, t]
        better = val > self.mx
        self.pred = torch.where(better, t, self.pred)
        self.mx = torch.where(better, val, self.mx)
        self.members.append(int(j))


class AvgAgg:
    """Probability-averaging committee.  P is (M, n, K), on the GPU or in host memory."""

    def __init__(self, P, dev):
        self.P = P
        self.dev = dev
        self.M, self.N, self.K = P.shape
        self.reset()

    def reset(self):
        self.S = torch.zeros((self.N, self.K), dtype=torch.float32, device=self.dev)
        self.pred = torch.zeros(self.N, dtype=torch.long, device=self.dev)
        self.members = []

    def _member(self, j):
        return self.P[j].to(self.dev, non_blocking=True).float()

    def trial(self, j):
        return (self.S + self._member(j)).argmax(dim=1)

    def add(self, j):
        self.S += self._member(j)
        self.pred = self.S.argmax(dim=1)
        self.members.append(int(j))


class RowAvgAgg(AvgAgg):
    """Probability averaging on a subset of rows; members are sliced when added."""

    def __init__(self, P, rows, dev):
        self.P = P
        self.rows = torch.as_tensor(np.asarray(rows), device=P.device)
        self.dev = dev
        self.M, self.K = P.shape[0], P.shape[2]
        self.N = len(rows)
        self.reset()

    def _member(self, j):
        return self.P[j][self.rows].to(self.dev, non_blocking=True).float()


class PluralityCounter:
    """Zero-one error of the plurality vote from its vote counts.

    Tracks, per point, the number of correct votes N and the largest number of
    votes for one wrong label m*.  With ties counted as errors the committee errs
    iff m* >= N (Theorem 4(i) of the paper); under a uniform random tie-break it
    errs with probability 1{m* > N} + 1{m* = N} r / (r + 1), r the number of wrong
    labels with m* votes, and ``errors(tie="random")`` returns that expectation.
    """

    def __init__(self, top1, y, K):
        self.T = top1
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=top1.device)
        self.M, self.N = top1.shape
        self.K = int(K)
        self.ar = torch.arange(self.N, device=top1.device)
        self.present = torch.unique(self.y)
        self.cls_n = torch.bincount(self.y, minlength=self.K)[self.present].double()
        self.reset()

    def reset(self):
        self.counts = torch.zeros((self.N, self.K), dtype=torch.int32, device=self.T.device)
        self.n_true = torch.zeros(self.N, dtype=torch.int32, device=self.T.device)
        self.m_star = torch.zeros(self.N, dtype=torch.int32, device=self.T.device)

    def add(self, j):
        t = self.T[j]
        wrong = t != self.y
        self.counts.index_put_((self.ar, t), torch.ones_like(t, dtype=torch.int32), accumulate=True)
        c = self.counts[self.ar, t]
        self.m_star = torch.where(wrong, torch.maximum(self.m_star, c), self.m_star)
        self.n_true = self.n_true + (~wrong).int()

    def pointwise(self, tie="error"):
        """Per-point error probability under ties as errors or a uniform random tie-break."""
        if tie == "error":
            return (self.m_star >= self.n_true).double()
        if tie != "random":
            raise ValueError(f"unknown tie rule {tie!r}")
        wrong_counts = self.counts.clone()
        wrong_counts[self.ar, self.y] = -1
        r = (wrong_counts == self.m_star[:, None]).sum(dim=1).double()
        tied = (self.m_star == self.n_true) & (self.n_true > 0)
        return ((self.m_star > self.n_true).double()
                + tied.double() * r / (r + 1.0))

    def errors(self, tie="error"):
        e = self.pointwise(tie)
        per_class = torch.bincount(self.y, weights=1.0 - e, minlength=self.K)[self.present] / self.cls_n
        return float(e.mean()), float(1.0 - per_class.mean())


# --------------------------------------------------------------------------- #
# paths and search
# --------------------------------------------------------------------------- #
def path_scores(agg, order, scorer, metrics):
    """Scores of every prefix of ``order``: {metric: [value at k = 1..len(order)]}."""
    agg.reset()
    out = {m: [] for m in metrics}
    for j in order:
        agg.add(j)
        for m in metrics:
            out[m].append(scorer(agg.pred, m))
    return out


def greedy(agg, scorer, metric, kmax=None, candidates=None):
    """Forward selection maximising ``metric``: (order, score after each addition)."""
    agg.reset()
    remaining = list(range(agg.M)) if candidates is None else list(candidates)
    kmax = len(remaining) if kmax is None else min(kmax, len(remaining))
    order, scores = [], []
    for _ in range(kmax):
        best_s, best_j = -np.inf, None
        for j in remaining:
            s = scorer(agg.trial(j), metric)
            if s > best_s:
                best_s, best_j = s, j
        agg.add(best_j)
        remaining.remove(best_j)
        order.append(best_j)
        scores.append(scorer(agg.pred, metric))
    return order, scores


def member_scores(top1, scorer, metric):
    return np.array([scorer(top1[i].long(), metric) for i in range(top1.shape[0])])
