"""A built pool, read for the Phase 3 analyses.

Everything here reads a pool directory written by build_pool.py and writes
nothing into it.  Two views of the stored logits are used.

* compact: per member and point, the top-1 label and its probability.  That is
  all a plurality vote, a confidence-weighted vote, per-point correctness,
  collision statistics and every label-free proposer need.  It is a few MB even
  for RxRx1, so it is cached under code/cache/phase3/<pool>/<split>.npz.
* probabilities: the full softmax, needed only for committees that average
  probabilities and for the Brier geometry.  It is built on demand and never
  cached, since for RxRx1 it is several GB per split.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(CODE, "cache", "phase3")


def pool_metric(K):
    return "gm" if K <= 10 else "macro_f1"


def paper_pool(name):
    """The pools of the paper's main analyses: three independent pools per benchmark and
    the refined pools.  Pools built for the frozen evaluation (replicas, public
    checkpoints) are analysed separately."""
    import re
    return re.fullmatch(r"(camelyon17|kather|iwildcam|rxrx1)(_convnext_atto|_mixed)?_seed_[123]"
                        r"|(camelyon17|kather|iwildcam|rxrx1)_refined_seed_101", name) is not None


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Pool:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.name = os.path.basename(self.root.rstrip("/\\"))
        with open(os.path.join(self.root, "pool_config.json"), encoding="utf-8") as f:
            self.config = json.load(f)["config"]
        with open(os.path.join(self.root, "predictions", "index.json"), encoding="utf-8") as f:
            self.index = json.load(f)
        rows = {}
        with open(os.path.join(self.root, "manifest.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["member_id"]] = r
        self.ids = [int(m) for m in self.index["accepted"]]
        self.rows = [rows[m] for m in self.ids]
        self.M = len(self.ids)
        self.K = int(self.index["num_classes"])
        self.benchmark = self.index["benchmark"]
        self.metric = pool_metric(self.K)
        self.splits = {s: m["role"] for s, m in self.index["splits"].items()}
        self.tuning = next((s for s, r in self.splits.items() if r == "tuning"), None)
        self.test = next((s for s, r in self.splits.items() if r == "test"), None)
        self.archs = [r["arch"] for r in self.rows]
        self._labels = {}

    # ----------------------------------------------------------------- members
    def single_arch(self):
        return len(set(self.archs)) == 1

    def source_val_score(self, metric=None):
        metric = metric or self.metric
        return np.array([r["metrics"]["source_val"][metric] for r in self.rows], dtype=float)

    def stored_score(self, split, metric=None):
        metric = metric or self.metric
        return np.array([r["metrics"][split][metric] for r in self.rows], dtype=float)

    def weight_stat(self, key):
        return np.array([(r.get("weight_stats") or {}).get(key, np.nan) for r in self.rows],
                        dtype=float)

    def schedule_modes(self):
        return [(r.get("schedule") or {}).get("mode", "recipe") for r in self.rows]

    # -------------------------------------------------------------------- data
    def labels(self, split):
        if split not in self._labels:
            path = os.path.join(self.root, "predictions", split, "labels.npy")
            self._labels[split] = np.load(path).astype(np.int64)
        return self._labels[split]

    def domains(self, split):
        return np.load(os.path.join(self.root, "predictions", split, "domains.npy"))

    def logit_path(self, split, member_id):
        return os.path.join(self.root, "predictions", split, f"member_{member_id:04d}.npy")

    def _softmax(self, split, member_id, dev):
        lg = np.load(self.logit_path(split, member_id))
        x = torch.from_numpy(np.ascontiguousarray(lg)).to(dev).float()
        return torch.softmax(x, dim=1)

    def compact(self, split, log=print):
        """{"top1": (M, N) int16, "conf": (M, N) float16}, cached."""
        path = os.path.join(CACHE, self.name, f"{split}.npz")
        if os.path.exists(path):
            z = np.load(path)
            if [int(v) for v in z["ids"]] == self.ids:
                return {"top1": z["top1"], "conf": z["conf"]}
        n = len(self.labels(split))
        top1 = np.empty((self.M, n), dtype=np.int16)
        conf = np.empty((self.M, n), dtype=np.float16)
        dev = device()
        t0 = time.time()
        for i, mid in enumerate(self.ids):
            p = self._softmax(split, mid, dev)
            c, t = p.max(dim=1)
            top1[i] = t.cpu().numpy().astype(np.int16)
            conf[i] = c.cpu().numpy().astype(np.float16)
            del p
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, ids=np.asarray(self.ids), top1=top1, conf=conf)
        log(f"    compact {self.name}/{split}: {self.M} x {n} in {time.time() - t0:.0f}s")
        return {"top1": top1, "conf": conf}

    def probs(self, split, rows=None, where="cuda"):
        """(M, n, K) float16 softmax probabilities, on the GPU or in host memory."""
        dev = device()
        n = len(self.labels(split)) if rows is None else len(rows)
        target = dev if where == "cuda" else torch.device("cpu")
        out = torch.empty((self.M, n, self.K), dtype=torch.float16, device=target)
        idx = None if rows is None else torch.as_tensor(np.asarray(rows), device=dev)
        for i, mid in enumerate(self.ids):
            p = self._softmax(split, mid, dev)
            if idx is not None:
                p = p[idx]
            out[i] = p.to(target, dtype=torch.float16)
            del p
        return out

    def probs_bytes(self, split, rows=None):
        n = len(self.labels(split)) if rows is None else len(rows)
        return 2 * self.M * n * self.K
