"""The pool as it stands, and pool_report.json.

The report is rewritten after every committed draw.  Its target-split numbers are
for watching a long build -- is the pool diverse, is the ensemble sane, how fast
is it going -- and nothing in the build reads them: acceptance uses source
validation alone (gate.py).  Analyses should recompute what they need from the
stored predictions rather than read it from here.

The ensemble and pairwise statistics need every accepted member's predictions,
so the running sums are held in memory and rebuilt from the prediction files
when a run resumes, rather than rewritten to disk after every member.
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np

from .metrics import decision_metrics, log_softmax, pairwise_stats, score_probs, spearman
from .store import utcnow

METRIC_KEYS = ("acc", "balanced_acc", "macro_f1", "gm", "brier", "nll", "ece", "confidence")


class PoolState:
    def __init__(self, labels, K):
        self.labels = {s: np.asarray(y, dtype=np.int64) for s, y in labels.items()}
        self.K = int(K)
        self.ids = []
        self.probsum, self.votes = {}, {}
        self.top1 = {s: [] for s in self.labels}

    def add(self, member_id, logits):
        for s, y in self.labels.items():
            lg = logits[s]
            if len(lg) != len(y):
                raise ValueError(f"member {member_id}, split {s}: {len(lg)} rows for {len(y)} labels")
            p = np.exp(log_softmax(lg))
            t = p.argmax(axis=1)
            if s not in self.probsum:
                self.probsum[s] = np.zeros(p.shape, dtype=np.float32)
                self.votes[s] = np.zeros(p.shape, dtype=np.int16)
            self.probsum[s] += p
            self.votes[s][np.arange(len(t)), t] += 1
            self.top1[s].append(t.astype(np.int32))
        self.ids.append(int(member_id))

    def rebuild(self, store, member_ids, log=print):
        t0 = time.time()
        for i, mid in enumerate(member_ids, 1):
            self.add(mid, {s: np.load(store.pred_path(s, mid)) for s in self.labels})
            if i % 10 == 0 or i == len(member_ids):
                log(f"  pool state rebuilt from {i}/{len(member_ids)} stored members "
                    f"({time.time() - t0:.0f}s)")


def _stats(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return None
    return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()),
            "median": float(np.median(v)), "max": float(v.max())}


def split_summary(state, rows, split, role, select_metric):
    labels = state.labels[split]
    members = [r for r in rows if r["status"] == "accepted" and split in r.get("metrics", {})]
    out = {"role": role, "n": int(len(labels)), "members": len(members),
           "member_metrics": {k: _stats([r["metrics"][split].get(k) for r in members])
                              for k in METRIC_KEYS}}
    M = len(state.ids)
    if M:
        out["ensemble_average"] = score_probs(state.probsum[split] / M, labels, state.K)
        out["ensemble_plurality"] = decision_metrics(state.votes[split].argmax(axis=1),
                                                     labels, state.K)
    if M >= 2:
        out.update(pairwise_stats(np.stack(state.top1[split]), labels))
        out["collision_floor"] = 1.0 / max(state.K - 1, 1)
    if split != "source_val" and len(members) >= 3:
        out["rank_corr_source_val_vs_this_split"] = spearman(
            [r["metrics"]["source_val"][select_metric] for r in members],
            [r["metrics"][split][select_metric] for r in members])
    return out


def member_entry(r):
    train = r.get("train") or {}
    return {"member_id": r["member_id"], "status": r["status"], "reasons": r.get("reasons", []),
            "parent": r.get("parent"), "recipe": r.get("recipe"),
            "best_epoch": train.get("best_epoch"), "epochs_run": train.get("epochs_run"),
            "minutes": round((r.get("seconds") or {}).get("total", 0.0) / 60.0, 2),
            "metrics": r.get("metrics", {}), "weight_stats": r.get("weight_stats")}


def build_report(meta, rows, state, split_meta):
    accepted = [r for r in rows if r["status"] == "accepted"]
    rejected = [r for r in rows if r["status"] != "accepted"]
    reasons = Counter(reason for r in rejected for reason in r.get("reasons", []))
    planned = meta["planned_members"]
    rate = len(accepted) / len(rows) if rows else None
    secs = [(r.get("seconds") or {}).get("total", 0.0) for r in rows]
    eta = None
    if rows and rate:
        eta = max(0, planned - len(accepted)) * float(np.mean(secs)) / rate / 3600.0

    summary = {s: split_summary(state, rows, s, split_meta[s]["role"], meta["select_metric"])
               for s in split_meta}

    warnings = []
    if len(rows) >= 10 and rate is not None and rate < 0.9:
        warnings.append(f"acceptance rate {rate:.2f}; see progress.rejection_reasons")
    if reasons.get("oom"):
        warnings.append("out-of-memory draws were rejected; fit the batch sizes with --preflight, "
                        "since a rejection caused by hardware conditions the pool on it")
    tuning = next((s for s, m in split_meta.items() if m["role"] == "tuning"),
                  next((s for s in split_meta if s != "source_val"), None))
    d = summary.get(tuning, {}).get("mean_pairwise_disagreement") if tuning else None
    if d is not None and len(accepted) >= 5 and d < 0.02:
        warnings.append(f"members disagree on only {d:.3f} of {tuning}; the recipe space may be too narrow")
    config_versions = {r.get("config_hash") for r in rows if r.get("config_hash")}
    if len(config_versions) > 1:
        warnings.append(f"recorded: this pool spans {len(config_versions)} configurations "
                        "(config_changes in pool_config.json, config_hash per member)")
    code_versions = {r.get("code_hash") for r in rows}
    if len(code_versions) > 1:
        warnings.append(f"members were built by {len(code_versions)} versions of the code; "
                        "see manifest code_hash")
    if meta.get("smoke"):
        warnings.append("SMOKE RUN: truncated training and evaluation; not a usable pool")

    return {
        "schema": "poolkit.report/1",
        "updated_utc": utcnow(),
        **meta,
        "note": ("Target-split numbers are monitoring only. Acceptance reads source validation "
                 "alone, and nothing in the build uses this report."),
        "progress": {"accepted": len(accepted), "planned": planned, "draws": len(rows),
                     "rejected": len(rejected), "acceptance_rate": rate,
                     "rejection_reasons": dict(reasons),
                     "mean_minutes_per_draw": (float(np.mean(secs)) / 60.0) if rows else None,
                     "eta_hours": eta, "complete": len(accepted) >= planned},
        "warnings": warnings,
        "summary": summary,
        "members": [member_entry(r) for r in rows],
    }
