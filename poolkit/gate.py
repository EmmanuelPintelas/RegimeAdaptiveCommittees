"""The acceptance gate: which trained draws enter the pool.

The policy, and why it is this and not a stricter one.

* The gate reads the member's own source validation predictions and nothing
  else.  A test that reads a target split would put target labels into the
  construction of the pool, and every later result on that split would be
  contaminated.  A test that compares a member with the members already
  accepted (a diversity or duplicate filter) would make members depend on each
  other, and the independence the theory assumes would no longer hold.  A test
  that is a function of one member's own source data keeps the accepted members
  independent: they are i.i.d. draws from the training process conditioned on
  passing.
* It rejects only draws that failed as training runs: non-finite outputs, a
  divergent loss, a run abandoned at chance, balanced accuracy on source
  validation below a small multiple of chance, or predictions collapsed onto one
  class.  It does not rank by quality.  A quality threshold on source validation
  would pre-apply the source-validation selection rule the paper compares
  against, and remove the weak-but-valid members whose presence is part of what
  makes committee selection matter.
* The thresholds live in the frozen pool configuration, identical in form across
  benchmarks and set relative to chance, and every rejected draw is committed to
  the manifest with its reasons, so acceptance rates are reported rather than
  hidden.
"""
from __future__ import annotations

import numpy as np

DEFAULTS = {
    # balanced accuracy on source validation must reach this multiple of chance,
    # chance being 1 / (number of classes present in source validation)
    "min_balanced_acc_x_chance": 1.5,
    # the most frequent predicted class may exceed the most frequent true class
    # share by at most this much
    "max_pred_share_excess": 0.4,
}


def resolve(cfg):
    out = dict(DEFAULTS)
    out.update(cfg or {})
    unknown = set(out) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"gate: unknown keys {sorted(unknown)}")
    if out["min_balanced_acc_x_chance"] < 1.0:
        raise ValueError("gate: min_balanced_acc_x_chance below 1 accepts worse than chance")
    return out


def apply_gate(gate, train_status, val_metrics, val_labels, finite):
    reasons, checks = [], {}
    if train_status != "trained":
        reasons.append(train_status)
    if not finite:
        reasons.append("nonfinite")
    if val_metrics:
        chance = 1.0 / max(int(val_metrics["k_present"]), 1)
        floor = gate["min_balanced_acc_x_chance"] * chance
        checks["balanced_acc"] = val_metrics["balanced_acc"]
        checks["balanced_acc_floor"] = floor
        if not np.isfinite(val_metrics["balanced_acc"]) or val_metrics["balanced_acc"] < floor:
            reasons.append("below_chance_floor")
        labels = np.asarray(val_labels, dtype=np.int64)
        true_share = float(np.bincount(labels).max() / len(labels))
        limit = min(0.999, true_share + gate["max_pred_share_excess"])
        checks["pred_share_max"] = val_metrics["pred_share_max"]
        checks["pred_share_limit"] = limit
        if val_metrics["pred_share_max"] > limit:
            reasons.append("collapsed")
    elif "nonfinite" not in reasons and train_status == "trained":
        reasons.append("no_validation")
    return {"passed": not reasons, "reasons": reasons, "checks": checks}
