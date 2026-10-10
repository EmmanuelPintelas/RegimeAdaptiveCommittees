"""The on-disk layout of a pool, and the writes that keep it consistent.

    <pool>/
      pool_config.json            frozen configuration, its hash, code hash, environment
      manifest.jsonl              one committed row per draw, accepted or rejected
      pool_report.json            rewritten after every commit
      events.jsonl                starts, resumes, interruptions, clean-ups
      models/member_0000.pt       checkpoints of accepted members
      predictions/
        index.json                splits, roles, sizes, dtype, accepted member ids
        <split>/labels.npy        int64 labels, row-aligned with every member file
        <split>/domains.npy       domain ids, where the benchmark provides them
        <split>/member_0000.npy   logits (N, K), float16
      rejected/
        predictions/<split>/member_0000.npy   rejected draws that did train
      logs/run_<time>.log

A member is committed when its manifest row is appended, and only then.  Its
files are written first, each under a temporary name and renamed into place, so
a run stopped at any point leaves either a committed member or files that the
next run recognises as uncommitted and deletes before training that slot again.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import re
import time

import numpy as np
import torch

MEMBER_RE = re.compile(r"^member_(\d+)\.")


class PoolLockedError(RuntimeError):
    pass


class DataMismatchError(RuntimeError):
    pass


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def clean(obj):
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def _replace(src, dst, tries=40, wait=0.25):
    # Windows refuses a rename onto a file another process has open (an editor
    # showing the report, a sync client); that clears within moments
    for attempt in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(wait)


def _atomic(path, write, binary):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    kwargs = {} if binary else {"encoding": "utf-8"}
    with open(tmp, "wb" if binary else "w", **kwargs) as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    _replace(tmp, path)


class FileLock:
    """An exclusive lock held for the life of the process; the OS frees it on exit."""

    def __init__(self, path):
        self.path = path
        self.fh = None

    def acquire(self):
        self.fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            self.fh = None
            raise PoolLockedError(
                f"another build_pool.py process is using {os.path.dirname(self.path)}") from None

    def release(self):
        if self.fh is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    self.fh.seek(0)
                    msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self.fh.close()
            self.fh = None


class PoolStore:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.config_path = os.path.join(self.root, "pool_config.json")
        self.manifest_path = os.path.join(self.root, "manifest.jsonl")
        self.report_path = os.path.join(self.root, "pool_report.json")
        self.events_path = os.path.join(self.root, "events.jsonl")
        self.lock_path = os.path.join(self.root, ".lock")
        self.models_dir = os.path.join(self.root, "models")
        self.pred_dir = os.path.join(self.root, "predictions")
        self.index_path = os.path.join(self.pred_dir, "index.json")
        self.rejected_dir = os.path.join(self.root, "rejected")
        self.logs_dir = os.path.join(self.root, "logs")

    # ------------------------------------------------------------------ paths
    def exists(self):
        return os.path.exists(self.config_path)

    def make_dirs(self):
        for d in (self.root, self.models_dir, self.pred_dir, self.logs_dir):
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def tag(member_id):
        return f"member_{int(member_id):04d}"

    def ckpt_path(self, member_id, rejected=False):
        base = os.path.join(self.rejected_dir, "models") if rejected else self.models_dir
        return os.path.join(base, self.tag(member_id) + ".pt")

    def pred_path(self, split, member_id, rejected=False):
        base = os.path.join(self.rejected_dir, "predictions") if rejected else self.pred_dir
        return os.path.join(base, split, self.tag(member_id) + ".npy")

    def labels_path(self, split):
        return os.path.join(self.pred_dir, split, "labels.npy")

    def domains_path(self, split):
        return os.path.join(self.pred_dir, split, "domains.npy")

    # ------------------------------------------------------------------ writes
    def write_json(self, path, obj):
        _atomic(path, lambda f: json.dump(clean(obj), f, indent=1), binary=False)

    @staticmethod
    def read_json(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def read_config(self):
        return self.read_json(self.config_path) if os.path.exists(self.config_path) else None

    def save_npy(self, path, arr):
        _atomic(path, lambda f: np.save(f, arr), binary=True)

    def save_torch(self, path, obj):
        _atomic(path, lambda f: torch.save(obj, f), binary=True)

    # ---------------------------------------------------------------- manifest
    def read_manifest(self):
        if not os.path.exists(self.manifest_path):
            return []
        with open(self.manifest_path, encoding="utf-8") as f:
            lines = f.read().split("\n")
        rows, torn = [], False
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if any(later.strip() for later in lines[i + 1:]):
                    raise RuntimeError(f"{self.manifest_path}: line {i + 1} is corrupt") from None
                torn = True
        if torn:
            # a crash during the append itself; the member was never committed
            _atomic(self.manifest_path,
                    lambda f: f.write("".join(json.dumps(r) + "\n" for r in rows)), binary=False)
            self.append_event("manifest_repaired", note="dropped a partially written last line")
        return rows

    def append_manifest(self, row):
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean(row)) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def append_event(self, kind, **info):
        os.makedirs(self.root, exist_ok=True)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean({"utc": utcnow(), "event": kind, **info})) + "\n")

    # ------------------------------------------------------------------ resume
    def cleanup(self, committed_ids):
        """Delete temporary files and every member file whose slot is not committed."""
        committed = {int(i) for i in committed_ids}
        removed = []
        for base in (self.models_dir, self.pred_dir, self.rejected_dir):
            if not os.path.isdir(base):
                continue
            for dirpath, _, files in os.walk(base):
                for fn in files:
                    m = MEMBER_RE.match(fn)
                    if fn.endswith(".tmp") or (m and int(m.group(1)) not in committed):
                        path = os.path.join(dirpath, fn)
                        os.remove(path)
                        removed.append(os.path.relpath(path, self.root))
        return removed

    def ensure_labels(self, split, labels, domains=None):
        """Write a split's labels once; on every later run, insist they are unchanged."""
        path = self.labels_path(split)
        if os.path.exists(path):
            old = np.load(path)
            if old.shape != labels.shape or not np.array_equal(old, labels):
                raise DataMismatchError(
                    f"split {split!r}: the labels on disk differ from the data now loaded "
                    f"({old.shape} vs {labels.shape}). The members already stored were "
                    "predicted on different data; point --data at the original dataset or "
                    "start a new pool.")
        else:
            self.save_npy(path, labels)
        if domains is not None and not os.path.exists(self.domains_path(split)):
            self.save_npy(self.domains_path(split), domains)
