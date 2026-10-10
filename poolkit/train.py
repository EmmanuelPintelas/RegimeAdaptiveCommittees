"""Training one pool member.

Built for a pool of a hundred members trained over days:

* seeding covers data order and augmentation workers as well as the weights, so a
  member interrupted and trained again from its slot is the same draw;
* the source-validation logits of the kept checkpoint are returned, so the stored
  source_val predictions cost no extra pass and belong to exactly that checkpoint;
* a loss that goes non-finite stops the run, which is reported as diverged
  instead of training on NaNs to the end;
* the learning-rate schedule is part of the recipe ("plateau",
  "halve_each_epoch" or "none") and is drawn and recorded like the rest.

Checkpoint selection reads source validation only.
"""
from __future__ import annotations

import gc
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from netkit.data import Transformed

from . import augment as A
from .metrics import score_logits

DEFAULT_TRAINING = {"abandon_patience": 5, "abandon_min_loss_drop": 0.01,
                    "two_stage_lr_prob": 0.0}


def two_stage_plan(epochs, bounds):
    """Train at the top of the learning-rate range, then at the bottom.

    The first ceil(epochs / 2) epochs run at the highest rate the member's range
    allows and the rest at the lowest, which is a coarse, explicit alternative to
    letting a plateau scheduler decide when to drop.  The rate drawn for the
    member is not used while this is on, and neither is its scheduler: a plateau
    rule moving the rate underneath a fixed two-stage plan would be neither.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    return {"mode": "two_stage", "high": hi, "low": lo,
            "switch_epoch": int(math.ceil(int(epochs) / 2))}


def should_abandon(streak, patience, loss_now, loss_at_streak_start, min_drop):
    """Is a run at chance also a dead run?

    A backbone that needs several epochs before its loss moves looks exactly
    like a dead run to a rule that reads validation alone, and on a split with
    a thousand classes the two are indistinguishable there for some time.  The
    training loss tells them apart: a dead run sits at log(K) while a slow one
    falls.  A run is therefore abandoned only when it has been at chance for
    ``patience`` epochs and its training loss has not fallen by ``min_drop``
    over that streak.
    """
    if streak < patience:
        return False
    if loss_at_streak_start is None or not np.isfinite(loss_at_streak_start) or loss_at_streak_start <= 0:
        return True
    return (loss_at_streak_start - loss_now) / loss_at_streak_start < min_drop


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    s = torch.utils.data.get_worker_info().seed % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


class HalveEachEpoch:
    """Halve the learning rate at the end of every epoch."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, *_):
        for group in self.optimizer.param_groups:
            group["lr"] *= 0.5


def make_optimizer(recipe, params):
    lr, wd = float(recipe["lr"]), float(recipe["weight_decay"])
    if recipe["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if recipe["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)


def make_scheduler(recipe, optimizer):
    if recipe["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=int(recipe["sched_patience"]),
            factor=float(recipe["sched_factor"]))
    if recipe["scheduler"] == "halve_each_epoch":
        return HalveEachEpoch(optimizer)
    return None


@torch.no_grad()
def predict_logits(model, loader, device, amp=True):
    """Logits for a whole loader, in order; a batch that overflows fp16 is redone in fp32."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    out, redone = [], 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        if not torch.isfinite(logits).all():
            logits = model(x).float()
            redone += 1
        out.append(logits.cpu().numpy())
    return np.concatenate(out), redone


def training_subset(base, frac, seed):
    if frac >= 1.0:
        return base
    n = len(base)
    k = max(1, int(round(n * frac)))
    idx = np.sort(np.random.default_rng(seed).choice(n, k, replace=False))
    return Subset(base, idx.tolist())


def train_member(model, recipe, seeds, train_base, val_loader, val_labels, K, select_metric,
                 device, *, image_size, workers, amp, aug_params, training=None,
                 lr_plan=None, max_batches=None, log=print):
    t_start = time.time()
    seed_everything(seeds["train"])
    ds = Transformed(training_subset(train_base, float(recipe["train_frac"]), seeds["data"]),
                     A.build_transform(recipe["augment"], image_size, aug_params))
    generator = torch.Generator()
    generator.manual_seed(seeds["data"])
    loader = DataLoader(ds, batch_size=int(recipe["batch_size"]), shuffle=True, drop_last=True,
                        num_workers=workers, pin_memory=device.type == "cuda",
                        persistent_workers=workers > 0, generator=generator,
                        worker_init_fn=seed_worker if workers > 0 else None)

    optimizer = make_optimizer(recipe, model.parameters())
    plan = lr_plan if (lr_plan or {}).get("mode") == "two_stage" else None
    scheduler = None if plan else make_scheduler(recipe, optimizer)
    if plan:
        log(f"    two-stage rate: {plan['high']:.2e} for epochs 1-{plan['switch_epoch']}, "
            f"then {plan['low']:.2e} (the drawn rate and scheduler are not used)")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(recipe["label_smoothing"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    epochs = int(recipe["epochs"])
    patience = max(1, int(recipe["stop_patience"]))
    chance = 1.0 / len(np.unique(val_labels))
    settings = {**DEFAULT_TRAINING, **(training or {})}
    abandon_patience = max(1, int(settings["abandon_patience"]))
    abandon_min_drop = float(settings["abandon_min_loss_drop"])

    best, history, status = None, [], "trained"
    since_best = at_chance = nonfinite = 0
    loss_at_streak_start = None
    try:
        for ep in range(1, epochs + 1):
            t_ep = time.time()
            if plan:
                target = plan["high"] if ep <= plan["switch_epoch"] else plan["low"]
                for group in optimizer.param_groups:
                    group["lr"] = target
            model.train()
            running, batches = 0.0, 0
            for bi, (x, y) in enumerate(loader):
                if max_batches is not None and bi >= max_batches:
                    break
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=scaler.is_enabled()):
                    loss = loss_fn(model(x), y)
                if not torch.isfinite(loss):
                    nonfinite += 1
                    if nonfinite >= 20:
                        status = "diverged"
                        break
                    continue
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += float(loss.detach())
                batches += 1
            if status == "diverged":
                log(f"    ep {ep:02d}: loss non-finite on {nonfinite} steps, stopped as diverged")
                break

            epoch_loss = running / max(1, batches)
            logits, _ = predict_logits(model, val_loader, device, amp)
            m = score_logits(logits, val_labels, K)
            score = m[select_metric]
            lr_now = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step(score)
            improved = best is None or score > best["score"]
            history.append({"epoch": ep, "train_loss": epoch_loss, "lr": lr_now,
                            "val": {k: m[k] for k in ("acc", "balanced_acc", "macro_f1", "gm",
                                                      "brier", "nll")},
                            "seconds": round(time.time() - t_ep, 1)})
            log(f"    ep {ep:02d}/{epochs}  loss {epoch_loss:.4f}  lr {lr_now:.1e}  "
                f"val acc {m['acc']:.4f}  {select_metric} {score:.4f}"
                f"{'  *' if improved else ''}  [{time.time() - t_ep:.0f}s]")
            if improved:
                best = {"score": score, "epoch": ep, "logits": logits, "metrics": m,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                since_best = 0
            else:
                since_best += 1
            if m["balanced_acc"] <= chance + 1e-6:
                if at_chance == 0:
                    loss_at_streak_start = epoch_loss
                at_chance += 1
            else:
                at_chance, loss_at_streak_start = 0, None
            if should_abandon(at_chance, abandon_patience, epoch_loss,
                              loss_at_streak_start, abandon_min_drop):
                status = "abandoned"
                log(f"    abandoned: at chance for {at_chance} epochs and the training loss "
                    f"fell by less than {abandon_min_drop:.0%} over them")
                break
            # under a two-stage plan the first stage is meant to run to its end,
            # so stopping is only considered once the rate has dropped
            if since_best >= patience and (plan is None or ep > plan["switch_epoch"]):
                log(f"    early stop after epoch {ep}")
                break
    finally:
        del loader
        gc.collect()

    result = {"status": status, "best_epoch": None, "epochs_run": len(history),
              "nonfinite_steps": nonfinite, "history": history, "val_logits": None,
              "val_metrics": None, "seconds": time.time() - t_start}
    if best is None:
        if status == "trained":
            result["status"] = "failed"
        return result
    model.load_state_dict(best["state"])
    model.eval()
    result.update(best_epoch=best["epoch"], val_logits=best["logits"], val_metrics=best["metrics"])
    return result
