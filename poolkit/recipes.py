"""Recipe distributions, and the draw that gives each member its recipe.

A pool is a sample from a training process, and the analyses built on it treat
members as independent draws from that process.  Three rules follow.

* The distribution is written down before the first member trains, and frozen
  with the pool; build_pool.py refuses to continue a pool under a changed one.
* A member's recipe and seeds are a function of (pool seed, slot) alone, never
  of what earlier members did.  No member depends on another, and an
  interrupted run redraws exactly the member it lost.
* Each parameter draws from its own random stream, keyed by its name, so
  widening one parameter's range does not silently change the others.

Streams are NumPy SeedSequence children, entropy = pool seed and spawn key =
(slot, purpose, parameter), which is the construction NumPy documents for
independent parallel streams.

A parameter is one of

    {"fixed": v}
    {"choice": [a, b, ...]}              optionally with "weights": [...]
    {"uniform": [lo, hi]}
    {"loguniform": [lo, hi]}
    {"int_uniform": [lo, hi]}            inclusive
    {"by": "arch", "cases": {"efficientnet_b0": <spec>, "convnext_atto": <spec>}}

The last draws from a different distribution depending on the value another
parameter took, e.g. a learning-rate range per architecture in a mixed pool.
The parameter it depends on must take finitely many values, every one of them
needs a case, and a case cannot itself be conditional.  The conditional draw
still comes from its own stream, so it is independent of everything but the
value it is conditioned on.
"""
from __future__ import annotations

import hashlib
import json
import math
import zlib

import numpy as np

KINDS = ("fixed", "choice", "uniform", "loguniform", "int_uniform", "by")

REQUIRED_BUILD = ("arch", "pretrained", "augment", "optimizer", "lr",
                  "weight_decay", "batch_size", "epochs", "scheduler",
                  "sched_patience", "sched_factor", "stop_patience",
                  "label_smoothing", "drop_rate", "train_frac")

# a refined member keeps its parent's architecture, weights and dropout
REQUIRED_REFINE = tuple(k for k in REQUIRED_BUILD
                        if k not in ("arch", "pretrained", "drop_rate"))

OPTIMIZERS = ("adam", "adamw", "sgd")
SCHEDULERS = ("plateau", "halve_each_epoch", "none")

_RECIPE, _SEEDS, _OPTION = 0, 1, 2


def _kind(spec):
    kinds = [k for k in KINDS if k in spec]
    if len(kinds) != 1:
        raise ValueError(f"expected exactly one of {KINDS}, got {sorted(spec)}")
    return kinds[0]


def support(spec):
    """The finite set of values a parameter can take, or None if continuous."""
    kind = _kind(spec)
    if kind == "fixed":
        return [spec["fixed"]]
    if kind == "choice":
        return list(spec["choice"])
    if kind == "int_uniform":
        lo, hi = spec["int_uniform"]
        return list(range(int(lo), int(hi) + 1))
    if kind == "by":
        values = []
        for case in spec["cases"].values():
            s = support(case)
            if s is None:
                return None
            values += [v for v in s if v not in values]
        return values
    return None


def _sequence(pool_seed, slot, purpose, key=0):
    return np.random.SeedSequence(entropy=int(pool_seed),
                                  spawn_key=(int(slot), int(purpose), int(key)))


def _stream(pool_seed, slot, name):
    return np.random.default_rng(_sequence(pool_seed, slot, _RECIPE,
                                           zlib.crc32(name.encode("utf-8"))))


def draw_value(spec, rng, drawn=None):
    kind = _kind(spec)
    if kind == "fixed":
        return spec["fixed"]
    if kind == "choice":
        options = spec["choice"]
        w = spec.get("weights")
        p = None if w is None else np.asarray(w, dtype=float) / float(np.sum(w))
        return options[int(rng.choice(len(options), p=p))]
    if kind == "uniform":
        lo, hi = spec["uniform"]
        return float(rng.uniform(lo, hi))
    if kind == "loguniform":
        lo, hi = spec["loguniform"]
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    if kind == "by":
        value = (drawn or {})[spec["by"]]
        case = spec["cases"].get(str(value))
        if case is None:
            raise ValueError(f"no case for {spec['by']}={value!r}")
        return draw_value(case, rng)
    lo, hi = spec["int_uniform"]
    return int(rng.integers(int(lo), int(hi) + 1))


def draw_recipe(space, pool_seed, slot):
    """The recipe of the member in ``slot``: one independent draw per parameter."""
    out = {name: draw_value(spec, _stream(pool_seed, slot, name))
           for name, spec in sorted(space.items()) if _kind(spec) != "by"}
    for name, spec in sorted(space.items()):
        if _kind(spec) == "by":
            out[name] = draw_value(spec, _stream(pool_seed, slot, name), out)
    return {name: out[name] for name in sorted(out)}


def draw_flag(prob, pool_seed, slot, name):
    """A per-member Bernoulli(prob) from its own stream.

    Used for options that are switched on for a fraction of the pool rather than
    drawn from a range, such as the two-stage learning-rate schedule.  The stream
    is separate from every parameter's, so turning an option on or off does not
    move any recipe draw, and a member's flag still depends only on the pool seed
    and its slot.
    """
    prob = float(prob)
    if prob <= 0.0:
        return False
    if prob >= 1.0:
        return True
    rng = np.random.default_rng(_sequence(pool_seed, slot, _OPTION,
                                          zlib.crc32(name.encode("utf-8"))))
    return bool(rng.random() < prob)


def value_bounds(spec, drawn=None):
    """The lowest and highest value a parameter can take, or None if unbounded.

    A conditional parameter is resolved with the values already drawn, so the
    bounds of a per-backbone learning rate are that backbone's own range.
    """
    kind = _kind(spec)
    if kind == "by":
        case = spec["cases"].get(str((drawn or {})[spec["by"]]))
        return None if case is None else value_bounds(case, drawn)
    if kind in ("uniform", "loguniform"):
        lo, hi = spec[kind]
        return float(lo), float(hi)
    values = support(spec) or []
    numbers = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return (min(numbers), max(numbers)) if numbers else None


def member_seeds(pool_seed, slot):
    """Seeds for weights and head initialisation, and for data order and augmentation."""
    state = _sequence(pool_seed, slot, _SEEDS).generate_state(2)
    return {"train": int(state[0]), "data": int(state[1])}


def _spec_errors(label, spec, allow_by):
    if not isinstance(spec, dict):
        return [f"{label}: expected an object like {{'choice': [...]}}"]
    try:
        kind = _kind(spec)
    except ValueError as exc:
        return [f"{label}: {exc}"]
    errors = []
    if kind == "by" and not allow_by:
        errors.append(f"{label}: a case cannot itself be conditional")
    if kind == "choice":
        if not spec["choice"]:
            errors.append(f"{label}: empty choice")
        w = spec.get("weights")
        if w is not None and (len(w) != len(spec["choice"]) or min(w) < 0 or sum(w) <= 0):
            errors.append(f"{label}: weights must match the choices and be non-negative")
    if kind in ("uniform", "loguniform", "int_uniform"):
        lo, hi = spec[kind]
        if not lo <= hi or (kind == "loguniform" and lo <= 0):
            errors.append(f"{label}: invalid range {spec[kind]}")
    return errors


def validate_space(space, mode, check_arch=None, check_augment=None):
    """Every problem with a recipe space, as a list of messages (empty if none)."""
    errors = []
    required = REQUIRED_REFINE if mode == "refine" else REQUIRED_BUILD
    for key in required:
        if key not in space:
            errors.append(f"recipe: missing parameter {key!r}")
    for name, spec in space.items():
        label = f"recipe.{name}"
        problems = _spec_errors(label, spec, allow_by=True)
        if problems:
            errors += problems
            continue
        if _kind(spec) == "by":
            ref, cases = spec["by"], spec.get("cases")
            if ref not in space:
                errors.append(f"{label}: depends on {ref!r}, which is not a parameter")
                continue
            ref_problems = _spec_errors(f"recipe.{ref}", space[ref], allow_by=True)
            if ref_problems or _kind(space[ref]) == "by":
                errors.append(f"{label}: {ref!r} must be a plain, valid parameter")
                continue
            ref_values = support(space[ref])
            if ref_values is None:
                errors.append(f"{label}: {ref!r} must take finitely many values")
                continue
            if not isinstance(cases, dict) or not cases:
                errors.append(f"{label}: 'cases' must map each value of {ref!r} to a spec")
                continue
            keys = {str(v) for v in ref_values}
            for v in ref_values:
                if str(v) not in cases:
                    errors.append(f"{label}: no case for {ref}={v!r}")
            for k, case in cases.items():
                if k not in keys:
                    errors.append(f"{label}: case {k!r} is not a value {ref!r} can take")
                errors += _spec_errors(f"{label}[{ref}={k}]", case, allow_by=False)
            if any(e.startswith(label) for e in errors):
                continue
        values = support(spec) or []
        checks = {"optimizer": lambda v: v in OPTIMIZERS,
                  "scheduler": lambda v: v in SCHEDULERS,
                  "arch": check_arch, "augment": check_augment}
        test = checks.get(name)
        if test is not None:
            for v in values:
                try:
                    ok = test(v)
                except Exception as exc:
                    ok, v = False, f"{v} ({exc})"
                if not ok:
                    errors.append(f"{label}: invalid value {v!r}")
    return errors


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def digest(obj, n=16):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()[:n]
