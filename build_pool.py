"""Build one pool of trained members, resumable member by member.

    python build_pool.py --config configs/pools/rxrx1.json --pool-seed 1
    python build_pool.py --out pools/rxrx1_seed_1                   (resume)
    python build_pool.py --config configs/pools/rxrx1.json --pool-seed 101 \\
                         --parent-pool pools/rxrx1_seed_1          (refined pool)
    python build_pool.py --out pools/rxrx1_seed_1 --status
    python build_pool.py --config configs/pools/rxrx1.json --pool-seed 1 --dry-run 20
    python build_pool.py --config configs/pools/rxrx1.json --preflight

Stop it at any time -- Ctrl+C, closing the window, a crash, a power cut.  The
member being trained is discarded, and the next run trains that slot again from
scratch with the same recipe and seeds and then continues to the planned size.
Every draw is committed to manifest.jsonl, pool_report.json is rewritten after
each one, and each member's logits on every split are stored as it is committed.
See poolkit/README.md for the layout and the policy behind it.
"""
from __future__ import annotations

import argparse
import copy
import gc
import glob
import hashlib
import json
import os
import platform
import sys
import time
import types
import warnings
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader, Subset  # noqa: E402

import paths  # noqa: E402
from poolkit import augment as A  # noqa: E402
from poolkit import gate as G  # noqa: E402
from poolkit import recipes as R  # noqa: E402
from poolkit import store as S  # noqa: E402
from poolkit import train as TR  # noqa: E402
from poolkit import weights as W  # noqa: E402
from poolkit.metrics import score_logits  # noqa: E402
from poolkit.report import PoolState, build_report  # noqa: E402
from netkit import data as D  # noqa: E402
from netkit import models as netkit_models  # noqa: E402
from netkit.data import Transformed  # noqa: E402

SELECT_METRICS = ("gm", "balanced_acc", "macro_f1", "acc")
CODE_FILES = ("build_pool.py", "poolkit/*.py", "netkit/models.py", "netkit/data.py")
GLOBAL_CONFIG = os.path.join(HERE, "configs", "global.json")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="benchmark configuration, configs/pools/<benchmark>.json; "
                                    "not needed to resume")
    p.add_argument("--out", help="pool directory; default <BUILD_POOLS>/<benchmark>_seed_<seed>")
    p.add_argument("--pool-seed", type=int, help="seed of this pool; pools of one benchmark "
                                                 "need different seeds to be independent")
    p.add_argument("--data", help="dataset root; default <paths.DATA>/<benchmark>")
    p.add_argument("--global-config", default=GLOBAL_CONFIG,
                   help="settings every pool inherits unless its own configuration overrides "
                        "them; read only when --config is given, since a resume uses the "
                        "configuration frozen into the pool")
    p.add_argument("--parent-pool", help="build a refined pool: members are fine-tuned children "
                                         "of this independently built pool's members")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--status", action="store_true", help="print progress and exit (no GPU)")
    p.add_argument("--dry-run", type=int, metavar="N",
                   help="print the recipes of the first N slots and the recipe distribution")
    p.add_argument("--preflight", action="store_true",
                   help="one training step at the largest batch size of every architecture")
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end check: 3 batches per epoch, 256 samples per split")
    p.add_argument("--stop-after", type=int, metavar="N",
                   help="exit cleanly after committing N draws in this run")
    p.add_argument("--allow-config-change", action="store_true",
                   help="continue under a configuration that differs from the frozen one; "
                        "the change is recorded in pool_config.json and events.jsonl")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def normalize(cfg):
    cfg.setdefault("mode", "build")
    cfg.setdefault("select_metric", "auto")
    cfg.setdefault("amp", True)
    cfg["gate"] = G.resolve(cfg.get("gate"))
    aug = dict(cfg.get("augment") or {})
    aug.setdefault("family_geometry", "flip")
    rf = copy.deepcopy(A.DEFAULT_RANDOM_FAMILY)
    rf.update(aug.get("random_family") or {})
    aug["random_family"] = rf
    cfg["augment"] = aug
    st = {"rejected_predictions": True, "rejected_checkpoints": False}
    st.update(cfg.get("store") or {})
    cfg["store"] = st
    if cfg["mode"] == "refine":
        ref = cfg.get("refine") or {}
        planned = int(ref.get("n_parents", 0)) * int(ref.get("children_per_parent", 0))
        cfg["pool_size"] = planned
        cfg["max_draws"] = planned
    return cfg


def merge_global(cfg, path):
    """Global defaults under the benchmark's own configuration.

    Dictionaries are merged one level deep and the benchmark's file wins, so a
    pool can override any default it needs to.  The merged result is what gets
    frozen and hashed, so a pool always records the settings it actually used
    rather than a reference to a file that may change later.
    """
    if not path or not os.path.exists(path):
        return cfg
    with open(path, encoding="utf-8") as f:
        defaults = json.load(f)
    defaults.pop("notes", None)
    merged = copy.deepcopy(defaults)
    for key, value in cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def science(cfg):
    return {k: v for k, v in cfg.items() if k != "notes"}


def validate(cfg, need_seed=True):
    import timm
    errors = []
    for key in ("benchmark", "image_size"):
        if key not in cfg:
            errors.append(f"missing {key!r}")
    if need_seed and not isinstance(cfg.get("pool_seed"), int):
        errors.append("pool_seed: give --pool-seed")
    if cfg["mode"] not in ("build", "refine"):
        errors.append(f"mode must be build or refine, got {cfg['mode']!r}")
    if not isinstance(cfg.get("pool_size"), int) or cfg["pool_size"] < 1:
        errors.append("pool_size must be a positive integer")
    if not isinstance(cfg.get("max_draws"), int) or cfg.get("max_draws", 0) < cfg.get("pool_size", 0):
        errors.append("max_draws must be an integer at least pool_size")
    if cfg["select_metric"] not in ("auto",) + SELECT_METRICS:
        errors.append(f"select_metric must be auto or one of {SELECT_METRICS}")
    unknown = set(cfg["store"]) - {"rejected_predictions", "rejected_checkpoints"}
    if unknown:
        errors.append(f"store: unknown keys {sorted(unknown)}")
    tr = cfg.get("training") or {}
    if not isinstance(tr, dict):
        errors.append("training: expected an object")
    else:
        unknown_tr = set(tr) - set(TR.DEFAULT_TRAINING)
        if unknown_tr:
            errors.append(f"training: unknown keys {sorted(unknown_tr)}; "
                          f"expected {sorted(TR.DEFAULT_TRAINING)}")
        if "abandon_patience" in tr and (not isinstance(tr["abandon_patience"], int)
                                         or tr["abandon_patience"] < 1):
            errors.append("training.abandon_patience must be a positive integer")
        if "abandon_min_loss_drop" in tr and not 0.0 <= float(tr["abandon_min_loss_drop"]) < 1.0:
            errors.append("training.abandon_min_loss_drop must be in [0, 1)")
        if "two_stage_lr_prob" in tr and not 0.0 <= float(tr["two_stage_lr_prob"]) <= 1.0:
            errors.append("training.two_stage_lr_prob must be in [0, 1]")
    if cfg["mode"] == "refine":
        space = (cfg.get("refine") or {}).get("recipe")
        if not space:
            errors.append("refine.recipe is missing")
    else:
        space = cfg.get("recipe")
        if not space:
            errors.append("recipe is missing")
    if space:
        errors += R.validate_space(space, cfg["mode"], check_arch=timm.is_model,
                                   check_augment=lambda n: A.check_setup(n, cfg["augment"]))
    return errors


def _flat(d, prefix=""):
    if not isinstance(d, dict):
        return {prefix[:-1]: d}
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}."))
    return out


def describe_diff(old, new):
    a, b = _flat(science(old)), _flat(science(new))
    return [f"{k}: {a.get(k, '<absent>')!r} -> {b.get(k, '<absent>')!r}"
            for k in sorted(set(a) | set(b)) if a.get(k, "<absent>") != b.get(k, "<absent>")]


def code_hash():
    h = hashlib.sha256()
    for pattern in CODE_FILES:
        for path in sorted(glob.glob(os.path.join(HERE, pattern))):
            h.update(os.path.relpath(path, HERE).replace("\\", "/").encode())
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:16]


def env_info():
    import timm
    import torchvision
    cuda = torch.cuda.is_available()
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "timm": timm.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if cuda else None}


def select_metric_for(name, K):
    if name != "auto":
        return name
    return "gm" if K <= 10 else "macro_f1"


def resolve_parents(cfg, runtime, frozen):
    """Fix, once, which members of the parent pool the refined pool descends from."""
    ref = cfg["refine"]
    old = ((frozen or {}).get("config") or {}).get("refine") or {}
    if old.get("resolved"):
        ref["resolved"] = old["resolved"]
    parent_dir = runtime.get("parent_pool")
    if not parent_dir:
        raise SystemExit("a refined pool needs --parent-pool")
    pstore = S.PoolStore(parent_dir)
    pconf = pstore.read_config()
    if pconf is None:
        raise SystemExit(f"{parent_dir} is not a pool built by build_pool.py")
    pc = pconf["config"]
    if pc["benchmark"] != cfg["benchmark"] or pc["image_size"] != cfg["image_size"]:
        raise SystemExit("the parent pool is for a different benchmark or image size")
    if pc.get("mode", "build") != "build":
        raise SystemExit("the parent must be an independently built pool, not a refined one")
    accepted = [r for r in pstore.read_manifest() if r["status"] == "accepted"]
    n = int(ref["n_parents"])
    if "resolved" not in ref:
        if len(accepted) < n:
            raise SystemExit(f"the parent pool has {len(accepted)} accepted members; {n} are needed")
        ref["resolved"] = {"parent_config_hash": pconf["config_hash"],
                           "parent_ids": [r["member_id"] for r in accepted[:n]]}
    else:
        if pconf["config_hash"] != ref["resolved"]["parent_config_hash"]:
            raise SystemExit("the parent pool's configuration changed after this refined pool began")
        have = {r["member_id"] for r in accepted}
        missing = [i for i in ref["resolved"]["parent_ids"] if i not in have]
        if missing:
            raise SystemExit(f"parent members {missing} are no longer in the parent pool")
    wanted = set(ref["resolved"]["parent_ids"])
    archs = sorted({r["recipe"]["arch"] for r in accepted if r["member_id"] in wanted})
    return pstore, archs


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def role_of(name):
    if name == "source_val":
        return "source_val"
    if name.startswith("ood_val"):
        return "tuning"
    if name.startswith("ood_test"):
        return "test"
    return "target"


def open_splits(bench, cfg):
    """Source validation and the targets; a smoke run takes a fixed random sample of each.

    A random sample and not the first rows: packed splits are often ordered by
    class, and the first rows of one would hold a single class.
    """
    limit = (cfg.get("smoke") or {}).get("eval_limit")
    splits = {"source_val": bench.source_val}
    splits.update(bench.targets)
    out = {}
    for name, ds in splits.items():
        if limit and len(ds) > limit:
            idx = np.sort(np.random.default_rng(0).choice(len(ds), limit, replace=False))
            ds = Subset(ds, idx.tolist())
        out[name] = ds
    return out


def labels_of(ds):
    base, idx = (ds.dataset, np.asarray(ds.indices)) if isinstance(ds, Subset) else (ds, None)
    y = getattr(base, "y", None)
    dom = getattr(base, "domains", None)
    if y is None:
        return np.asarray([ds[i][1] for i in range(len(ds))], dtype=np.int64), None
    y = np.asarray(y, dtype=np.int64)
    dom = None if dom is None else np.asarray(dom)
    if idx is not None:
        y = y[idx]
        dom = None if dom is None else dom[idx]
    return y, dom


# --------------------------------------------------------------------------- #
# inspection commands
# --------------------------------------------------------------------------- #
def lr_plan_for(cfg, space, recipe, slot):
    """The two-stage learning-rate plan of this member, or None.

    The flag is a per-member draw at the pool-wide probability, and the two rates
    are the ends of that member's own range, which in a mixed pool means its
    backbone's range.  Refined pools are excluded: their members are short
    fine-tuning passes whose rate is the point of the draw.
    """
    if cfg["mode"] == "refine":
        return None
    prob = float((cfg.get("training") or {}).get(
        "two_stage_lr_prob", TR.DEFAULT_TRAINING["two_stage_lr_prob"]))
    if not R.draw_flag(prob, cfg["pool_seed"], slot, "two_stage_lr"):
        return None
    bounds = R.value_bounds(space["lr"], recipe)
    if bounds is None or not bounds[1] > bounds[0]:
        return None
    return TR.two_stage_plan(int(recipe["epochs"]), bounds)


def recipe_line(r):
    parts = []
    for key, fmt in (("arch", "{:22}"), ("augment", "{:16}"), ("optimizer", "{:5}"),
                     ("lr", "lr={:.1e}"), ("weight_decay", "wd={:g}"), ("batch_size", "bs={}"),
                     ("epochs", "ep={}"), ("scheduler", "{}"), ("label_smoothing", "ls={:g}"),
                     ("drop_rate", "dr={:g}")):
        if key in r:
            parts.append(fmt.format(r[key]))
    return " ".join(parts)


def cmd_status(store):
    conf = store.read_config()
    if conf is None:
        print(f"no pool at {store.root}")
        return 1
    c = conf["config"]
    rows = store.read_manifest()
    acc = [r for r in rows if r["status"] == "accepted"]
    reasons = Counter(x for r in rows if r["status"] != "accepted" for x in r.get("reasons", []))
    print(f"{store.root}\n  {c['benchmark']}  mode {c['mode']}  pool seed {c['pool_seed']}  "
          f"config {conf['config_hash']}")
    print(f"  accepted {len(acc)}/{c['pool_size']}  draws {len(rows)}  "
          f"rejected {len(rows) - len(acc)} {dict(reasons) if reasons else ''}")
    for r in rows[-8:]:
        m = r.get("metrics", {}).get("source_val", {})
        print(f"    [{r['member_id']:04d}] {r['status']:8} {recipe_line(r['recipe'])}  "
              f"src acc {m.get('acc', float('nan')):.4f}  "
              f"{(r.get('seconds') or {}).get('total', 0) / 60:.1f} min")
    if os.path.exists(store.report_path):
        rep = store.read_json(store.report_path)
        eta = rep["progress"].get("eta_hours")
        print(f"  mean {rep['progress'].get('mean_minutes_per_draw') or 0:.1f} min per draw"
              + (f", about {eta:.1f} h to go" if eta is not None else ""))
        sel = rep.get("select_metric")
        for split, s in rep.get("summary", {}).items():
            ens = (s.get("ensemble_average") or {}).get(sel)
            mm = (s.get("member_metrics") or {}).get(sel) or {}
            coll = s.get("collision_probability")
            print(f"    {split:22} members {sel} mean {mm.get('mean', float('nan')):.4f}  "
                  f"ensemble {ens if ens is None else f'{ens:.4f}'}  "
                  f"disagreement {s.get('mean_pairwise_disagreement')}  collision {coll}")
        for w in rep.get("warnings", []):
            print(f"  WARNING {w}")
    return 0


def cmd_dry_run(cfg, n):
    refine = cfg["mode"] == "refine"
    space = cfg["refine"]["recipe"] if refine else cfg["recipe"]
    print(f"{cfg['benchmark']}  mode {cfg['mode']}  pool seed {cfg['pool_seed']}  "
          f"planned {cfg['pool_size']}  max draws {cfg['max_draws']}")
    for slot in range(n):
        r = R.draw_recipe(space, cfg["pool_seed"], slot)
        plan = lr_plan_for(cfg, space, r, slot)
        extra = (f"  two-stage {plan['high']:.1e}->{plan['low']:.1e} @ep{plan['switch_epoch']}"
                 if plan else "")
        if refine:
            c = int(cfg["refine"]["children_per_parent"])
            ids = cfg["refine"]["resolved"]["parent_ids"]
            extra += f"  parent {ids[slot // c]}" if slot // c < len(ids) else "  (beyond plan)"
        print(f"  [{slot:04d}] {recipe_line(r)}{extra}")
    draws = [R.draw_recipe(space, cfg["pool_seed"], s) for s in range(1000)]
    print("\ndistribution over the first 1000 slots")
    for name, spec in sorted(space.items()):
        values = [d[name] for d in draws]
        if R.support(spec) is not None:
            counts = Counter(values)
            print(f"  {name:16} " + ", ".join(f"{k}: {v / 10:.1f}%" for k, v in counts.most_common()))
        else:
            v = np.asarray(values, dtype=float)
            print(f"  {name:16} min {v.min():.3g}  median {np.median(v):.3g}  max {v.max():.3g}")
    return 0


def cmd_preflight(cfg, K, archs, device, eval_bs):
    if device.type != "cuda":
        print("no GPU visible; nothing to check")
        return 0
    space = cfg["refine"]["recipe"] if cfg["mode"] == "refine" else cfg["recipe"]
    batches = R.support(space["batch_size"])
    bs, size, ok = int(max(batches)), int(cfg["image_size"]), True
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    print(f"preflight: batch {bs}, image {size}, {K} classes, eval batch {eval_bs}")
    for arch in archs:
        model = netkit_models.build(arch, K, False, 0.0, device, size)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["amp"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model.train()
            for _ in range(2):
                x = torch.randn(bs, 3, size, size, device=device)
                y = torch.randint(0, K, (bs,), device=device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=bool(cfg["amp"])):
                    loss = torch.nn.functional.cross_entropy(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", enabled=bool(cfg["amp"])):
                model(torch.randn(eval_bs, 3, size, size, device=device))
            peak = torch.cuda.max_memory_allocated() / 2 ** 20
            print(f"  {arch:24} ok    peak {peak:,.0f} MiB of {total:,.0f} MiB")
        except torch.cuda.OutOfMemoryError:
            ok = False
            print(f"  {arch:24} OUT OF MEMORY at batch {bs}: remove that batch size or architecture")
        finally:
            del model, opt
            gc.collect()
            torch.cuda.empty_cache()
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
class Tee:
    def __init__(self, path, stream):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "a", encoding="utf-8", buffering=1)
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def run_slot(slot, ctx):
    cfg, store = ctx.cfg, ctx.store
    refine = cfg["mode"] == "refine"
    space = cfg["refine"]["recipe"] if refine else cfg["recipe"]
    recipe = R.draw_recipe(space, cfg["pool_seed"], slot)
    seeds = R.member_seeds(cfg["pool_seed"], slot)
    parent = None
    if refine:
        group = slot // int(cfg["refine"]["children_per_parent"])
        resolved = cfg["refine"]["resolved"]
        parent = {"member_id": resolved["parent_ids"][group], "group": group,
                  "pool_config_hash": resolved["parent_config_hash"]}
    started = S.utcnow()
    t0 = time.time()
    plan = lr_plan_for(cfg, space, recipe, slot)
    print(f"\n[{slot:04d}] {recipe_line(recipe)}"
          + (f"  <- parent {parent['member_id']}" if parent else "")
          + (f"  two-stage {plan['high']:.1e}->{plan['low']:.1e} @ep{plan['switch_epoch']}"
             if plan else ""))

    model = None
    try:
        TR.seed_everything(seeds["train"])
        if refine:
            model = netkit_models.load_member(ctx.parent_store.ckpt_path(parent["member_id"]), ctx.device)
            recipe_arch = model.arch
        else:
            model = netkit_models.build(recipe["arch"], ctx.K, bool(recipe["pretrained"]),
                                       float(recipe["drop_rate"]), ctx.device, cfg["image_size"])
            recipe_arch = recipe["arch"]
        outcome = TR.train_member(
            model, recipe, seeds, ctx.train_base, ctx.val_loader, ctx.labels["source_val"],
            ctx.K, ctx.select_metric, ctx.device, image_size=cfg["image_size"],
            workers=ctx.runtime["workers"], amp=cfg["amp"], aug_params=cfg["augment"],
            training=cfg.get("training"), lr_plan=plan,
            max_batches=(cfg.get("smoke") or {}).get("train_batches"))
    except torch.cuda.OutOfMemoryError:
        print("    out of memory; recorded as a rejected draw (fit batch sizes with --preflight)")
        model, recipe_arch = None, recipe.get("arch")
        outcome = {"status": "oom", "best_epoch": None, "epochs_run": 0, "nonfinite_steps": 0,
                   "history": [], "val_logits": None, "val_metrics": None}
    torch.cuda.empty_cache()
    t_train = time.time() - t0

    status = outcome["status"]
    finite = True
    if model is not None and outcome["val_logits"] is not None:
        finite = bool(np.isfinite(outcome["val_logits"]).all()) and all(
            bool(torch.isfinite(v).all()) for v in model.state_dict().values() if v.is_floating_point())
    verdict = G.apply_gate(cfg["gate"], status, outcome["val_metrics"], ctx.labels["source_val"], finite)
    accepted = verdict["passed"]

    metrics, logits = {}, {}
    if outcome["val_metrics"] is not None:
        metrics["source_val"] = outcome["val_metrics"]
    row = {"member_id": slot, "status": "accepted" if accepted else "rejected",
           "reasons": verdict["reasons"], "mode": cfg["mode"], "arch": recipe_arch,
           "recipe": recipe, "seeds": seeds, "parent": parent, "gate": verdict["checks"],
           "train": {k: outcome[k] for k in ("status", "best_epoch", "epochs_run",
                                             "nonfinite_steps", "history")},
           "schedule": plan or {"mode": "recipe", "scheduler": recipe["scheduler"]},
           "code_hash": ctx.code_hash, "config_hash": ctx.config_hash, "started_utc": started}

    t1 = time.time()
    keep = model is not None and finite and (
        accepted or (status == "trained" and cfg["store"]["rejected_predictions"]))
    if keep:
        logits["source_val"] = outcome["val_logits"]
        joint, redone = TR.predict_logits(model, ctx.target_loader, ctx.device, cfg["amp"])
        for name, (a, b) in ctx.target_offsets.items():
            logits[name] = joint[a:b]
        dtypes = {}
        for name, lg in logits.items():
            if name != "source_val":
                metrics[name] = score_logits(lg, ctx.labels[name], ctx.K)
            dtype = np.float16 if float(np.abs(lg).max()) < 6.0e4 else np.float32
            store.save_npy(store.pred_path(name, slot, rejected=not accepted), lg.astype(dtype))
            dtypes[name] = np.dtype(dtype).name
        row["pred_dtype"] = dtypes
        row["fp32_redone_batches"] = redone
        row["weight_stats"] = W.weight_stats(model.state_dict())
    if model is not None and status == "trained" and (accepted or cfg["store"]["rejected_checkpoints"]):
        payload = {"arch": model.arch, "num_classes": model.num_classes,
                   "image_size": model.image_size,
                   "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   "member_id": slot, "recipe": recipe, "seeds": seeds, "parent": parent,
                   "pool_config_hash": ctx.config_hash, "val_metrics": outcome["val_metrics"]}
        store.save_torch(store.ckpt_path(slot, rejected=not accepted), payload)

    row["metrics"] = metrics
    row["seconds"] = {"train": round(t_train, 1), "predict_and_store": round(time.time() - t1, 1),
                      "total": round(time.time() - t0, 1)}
    row["finished_utc"] = S.utcnow()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return row, (logits if accepted else None)


def write_outputs(ctx, rows, state):
    cfg = ctx.cfg
    parent = None
    if cfg["mode"] == "refine":
        parent = {"pool": ctx.runtime.get("parent_pool"),
                  "children_per_parent": cfg["refine"]["children_per_parent"],
                  **cfg["refine"]["resolved"]}
    meta = {"benchmark": cfg["benchmark"], "pool_dir": ctx.store.root, "mode": cfg["mode"],
            "pool_seed": cfg["pool_seed"], "config_hash": ctx.config_hash,
            "code_hash": ctx.code_hash, "num_classes": ctx.K, "select_metric": ctx.select_metric,
            "planned_members": cfg["pool_size"], "max_draws": cfg["max_draws"],
            "smoke": bool(cfg.get("smoke")), "parent": parent}
    report = build_report(meta, rows, state, ctx.split_meta)
    ctx.store.write_json(ctx.store.report_path, report)
    ctx.store.write_json(ctx.store.index_path, {
        "schema": "poolkit.predictions/1", "benchmark": cfg["benchmark"],
        "num_classes": ctx.K, "image_size": cfg["image_size"],
        "content": ("pre-softmax logits of the kept checkpoint, one file per member per split, "
                    "rows aligned with that split's labels.npy; float16 unless the manifest row's "
                    "pred_dtype says otherwise"),
        "splits": {s: {"role": m["role"], "n": m["n"], "labels": f"{s}/labels.npy",
                       "domains": f"{s}/domains.npy" if m["has_domains"] else None}
                   for s, m in ctx.split_meta.items()},
        "accepted": [r["member_id"] for r in rows if r["status"] == "accepted"],
        "rejected_with_predictions": [r["member_id"] for r in rows
                                      if r["status"] != "accepted" and r.get("pred_dtype")],
        "rejected_predictions_dir": "../rejected/predictions",
    })
    return report


def build(args, cfg, runtime, frozen, store, parent_store):
    chash = code_hash()
    config_hash = R.digest(science(cfg))
    public_runtime = {k: runtime.get(k) for k in ("data_root", "parent_pool")}
    if frozen is None:
        store.write_json(store.config_path, {
            "schema": "poolkit.pool_config/1", "created_utc": S.utcnow(),
            "config_hash": config_hash, "config": cfg, "runtime": public_runtime,
            "code_hash": chash, "env": env_info()})
        store.append_event("created", config_hash=config_hash, code_hash=chash)
    else:
        if frozen["config_hash"] != config_hash:
            changes = describe_diff(frozen["config"], cfg)
            if not args.allow_config_change:
                print("the configuration differs from the one this pool was started with:")
                for line in changes:
                    print(f"  {line}")
                print("start a new pool, restore the configuration, or pass --allow-config-change")
                return 2
            frozen.setdefault("config_changes", []).append(
                {"utc": S.utcnow(), "from": frozen["config_hash"], "to": config_hash,
                 "changes": changes})
            frozen["config"], frozen["config_hash"] = cfg, config_hash
            store.append_event("config_changed", changes=changes)
        if frozen.get("code_hash") != chash:
            store.append_event("code_changed", previous=frozen.get("code_hash"), now=chash)
            print(f"note: the code differs from the version that started this pool "
                  f"({frozen.get('code_hash')} -> {chash}); each manifest row records its own")
        frozen["runtime"] = public_runtime
        store.write_json(store.config_path, frozen)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"pool      {store.root}")
    print(f"config    {config_hash}  code {chash}  mode {cfg['mode']}  seed {cfg['pool_seed']}")
    print(f"device    {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    bench = D.load(cfg["benchmark"], runtime["data_root"])
    K = bench.num_classes
    select = select_metric_for(cfg["select_metric"], K)
    splits = open_splits(bench, cfg)
    labels, split_meta = {}, {}
    for name, ds in splits.items():
        y, dom = labels_of(ds)
        store.ensure_labels(name, y, dom)
        labels[name] = y
        split_meta[name] = {"role": role_of(name), "n": int(len(y)), "has_domains": dom is not None}
    if len(np.unique(labels["source_val"])) < 2:
        print("source validation holds a single class; the gate and checkpoint selection "
              "need at least two")
        return 2
    print(f"data      {cfg['benchmark']}: {K} classes, selection metric {select}, source train "
          f"{len(bench.source_train)}; " + ", ".join(f"{s} {m['n']}" for s, m in split_meta.items()))

    rows = store.read_manifest()
    committed = {int(r["member_id"]) for r in rows}
    removed = store.cleanup(committed)
    if removed:
        print(f"resume    removed {len(removed)} file(s) of an uncommitted member")
        store.append_event("cleanup", removed=removed)

    ctx = types.SimpleNamespace(cfg=cfg, store=store, runtime=runtime, device=device, K=K,
                                select_metric=select, labels=labels, split_meta=split_meta,
                                code_hash=chash, config_hash=config_hash,
                                parent_store=parent_store, train_base=bench.source_train)
    accepted_ids = [int(r["member_id"]) for r in rows if r["status"] == "accepted"]
    state = PoolState(labels, K)
    if accepted_ids:
        print(f"resume    {len(accepted_ids)} accepted and {len(rows) - len(accepted_ids)} "
              f"rejected draws on record")
        state.rebuild(store, accepted_ids)
    write_outputs(ctx, rows, state)
    store.append_event("run_start", committed=len(rows), accepted=len(accepted_ids),
                       data_root=runtime["data_root"], workers=runtime["workers"])

    planned, max_draws = cfg["pool_size"], cfg["max_draws"]
    n_acc = len(accepted_ids)
    if n_acc >= planned:
        print(f"pool complete: {n_acc}/{planned} members")
        return 0

    eval_tf = A.eval_transform(cfg["image_size"])
    workers, pin = runtime["workers"], device.type == "cuda"
    ctx.val_loader = DataLoader(Transformed(splits["source_val"], eval_tf),
                                batch_size=runtime["eval_batch_size"], shuffle=False,
                                num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    parts, ctx.target_offsets, start = [], {}, 0
    for name, ds in splits.items():
        if name == "source_val":
            continue
        parts.append(Transformed(ds, eval_tf))
        ctx.target_offsets[name] = (start, start + len(ds))
        start += len(ds)
    ctx.target_loader = DataLoader(ConcatDataset(parts), batch_size=runtime["eval_batch_size"],
                                   shuffle=False, num_workers=workers, pin_memory=pin,
                                   persistent_workers=workers > 0)

    crash_at = os.environ.get("POOLKIT_TEST_CRASH_BEFORE_COMMIT")
    slot, commits = 0, 0
    try:
        while n_acc < planned and slot < max_draws:
            if slot in committed:
                slot += 1
                continue
            row, logits = run_slot(slot, ctx)
            if crash_at is not None and int(crash_at) == slot:
                print("TEST HOOK: exiting after writing files, before the commit")
                sys.stdout.flush()
                os._exit(3)
            store.append_manifest(row)
            rows.append(row)
            committed.add(slot)
            if row["status"] == "accepted":
                state.add(slot, logits)
                n_acc += 1
            report = write_outputs(ctx, rows, state)
            sel = {s: m[select] for s, m in row["metrics"].items() if select in m}
            eta = report["progress"]["eta_hours"]
            print(f"    {row['status']}"
                  + (f" ({', '.join(row['reasons'])})" if row["reasons"] else "")
                  + "  " + "  ".join(f"{s} {v:.4f}" for s, v in sel.items())
                  + f"  | {row['seconds']['total'] / 60:.1f} min | pool {n_acc}/{planned}"
                  + (f" | ~{eta:.1f} h left" if eta is not None else ""))
            commits += 1
            slot += 1
            if args.stop_after and commits >= args.stop_after:
                store.append_event("stopped", reason="--stop-after", commits=commits)
                print(f"\nstopped after {commits} draw(s) as asked; run again to continue")
                return 0
    except KeyboardInterrupt:
        store.append_event("interrupted", slot=slot)
        store.cleanup(committed)
        print(f"\ninterrupted while training slot {slot}; it is discarded and will be trained "
              f"again on resume. {n_acc}/{planned} members are committed.")
        return 130

    if n_acc < planned:
        store.append_event("max_draws_exhausted", accepted=n_acc, draws=len(rows))
        print(f"\nmax_draws {max_draws} reached with {n_acc}/{planned} members accepted; "
              "see the rejection reasons in pool_report.json")
        return 4
    store.append_event("complete", accepted=n_acc, draws=len(rows))
    print(f"\npool complete: {n_acc}/{planned} members from {len(rows)} draws")
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    warnings.filterwarnings("ignore", message=".*y_pred contains classes.*")

    file_cfg = None
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            file_cfg = json.load(f)
        file_cfg = merge_global(file_cfg, args.global_config)
    if args.out:
        out = os.path.abspath(args.out)
    elif file_cfg is not None and (args.pool_seed is not None or "pool_seed" in file_cfg):
        seed = args.pool_seed if args.pool_seed is not None else file_cfg["pool_seed"]
        name = f"{file_cfg['benchmark']}_{'refined_' if args.parent_pool else ''}seed_{seed}"
        out = os.path.join(paths.BUILD_POOLS, ("smoke_" if args.smoke else "") + name)
    else:
        out = None

    if args.status:
        if out is None:
            raise SystemExit("--status needs --out")
        return cmd_status(S.PoolStore(out))

    store = S.PoolStore(out) if out else None
    frozen = store.read_config() if store is not None else None
    if file_cfg is None and frozen is None:
        raise SystemExit("nothing to do: no pool at --out, and no --config to start one")
    cfg = copy.deepcopy(file_cfg if file_cfg is not None else frozen["config"])
    if args.pool_seed is not None:
        cfg["pool_seed"] = args.pool_seed
    elif frozen is not None and "pool_seed" not in cfg:
        cfg["pool_seed"] = frozen["config"]["pool_seed"]
    if args.parent_pool or (frozen and frozen["config"].get("mode") == "refine"):
        cfg["mode"] = "refine"
    if args.smoke:
        cfg["smoke"] = {"train_batches": 3, "eval_limit": 256}
    elif frozen and frozen["config"].get("smoke"):
        cfg["smoke"] = frozen["config"]["smoke"]
    previous = (frozen or {}).get("runtime") or {}
    runtime = {"data_root": args.data or previous.get("data_root") or cfg.get("data_root")
               or os.path.join(paths.DATA, cfg["benchmark"]),
               "parent_pool": (os.path.abspath(args.parent_pool) if args.parent_pool
                               else previous.get("parent_pool")),
               "workers": args.workers, "eval_batch_size": args.eval_batch_size}
    cfg.pop("data_root", None)
    normalize(cfg)

    parent_store, parent_archs = None, None
    if cfg["mode"] == "refine":
        parent_store, parent_archs = resolve_parents(cfg, runtime, frozen)

    errors = validate(cfg, need_seed=not args.preflight)
    if errors:
        print("configuration problems:")
        for e in errors:
            print(f"  {e}")
        return 2

    if args.dry_run:
        return cmd_dry_run(cfg, args.dry_run)

    if args.preflight:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        K = D.load(cfg["benchmark"], runtime["data_root"]).num_classes
        archs = parent_archs if parent_archs else R.support(cfg["recipe"]["arch"])
        return cmd_preflight(cfg, K, archs, device, runtime["eval_batch_size"])

    if store is None:
        raise SystemExit("give --pool-seed (to start a pool) or --out (to resume one)")
    store.make_dirs()
    lock = S.FileLock(store.lock_path)
    try:
        lock.acquire()
    except S.PoolLockedError as exc:
        print(exc)
        return 3
    tee = Tee(os.path.join(store.logs_dir, time.strftime("run_%Y%m%d_%H%M%S.log")), sys.stdout)
    sys.stdout = tee
    try:
        return build(args, cfg, runtime, frozen, store, parent_store)
    finally:
        sys.stdout = tee.stream
        tee.close()
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
