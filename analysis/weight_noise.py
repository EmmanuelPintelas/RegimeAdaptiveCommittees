"""Weight-noise stability of every member, from unlabeled target inputs (proposer P8).

For each member: the share of a fixed set of unlabeled test images whose predicted
class changes when every parameter tensor receives Gaussian noise scaled to that
tensor's own standard deviation, averaged over a few draws, at each noise scale.
Batch-norm statistics are buffers, not parameters, and are left alone.  No label is
read.  Cached under code/cache/phase3/<pool>/weight_noise.json.

    python -m analysis.weight_noise --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .pools import CACHE, Pool, device, paper_pool

SIGMAS = (0.02, 0.05, 0.1)
POINTS = 2000
DRAWS = 2
SEED = 0


@torch.no_grad()
def perturbed_argmax(model, X, dev, sigma, gen, bs=256):
    """Predicted classes after N(0, sigma * std(W)) noise on every parameter tensor W."""
    saved = []
    for p in (model.parameters() if sigma > 0 else ()):
        if p.is_floating_point() and p.dim() > 0:
            saved.append((p, p.detach().clone()))
            sd = float(p.detach().float().std())
            if sd > 0:
                p.data.add_((torch.randn(p.shape, generator=gen, device=p.device) * (sigma * sd))
                            .to(p.dtype))
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            out.append(model(X[i:i + bs].to(dev, non_blocking=True)).float().argmax(1).cpu())
    for p, orig in saved:
        p.data.copy_(orig)
    return torch.cat(out).numpy()


def target_images(pool, rows, workers=2):
    """The pool's test images at ``rows``, transformed for evaluation; labels unused."""
    import paths
    from netkit import data as D
    from netkit.augment import eval_transform
    from netkit.data import Transformed
    bench = D.load(pool.benchmark, os.path.join(paths.DATA, pool.benchmark))
    ds = Transformed(Subset(bench.targets[pool.test], rows.tolist()),
                     eval_transform(pool.config["image_size"]))
    return torch.cat([x for x, _ in DataLoader(ds, batch_size=256, shuffle=False,
                                                 num_workers=workers)])


def cache_path(pool, variant=""):
    return os.path.join(CACHE, pool.name, f"weight_noise{'_' + variant if variant else ''}.json")


def candidate_rows(pool, variant=""):
    """Rows the sample is drawn from: the whole test split, or for the inductive check its
    proposer half (whole test domains, the half the proposers may read)."""
    n = np.load(pool.logit_path(pool.test, pool.ids[0]), mmap_mode="r").shape[0]
    if variant != "inductive":
        return np.arange(n)
    from .method import domain_halves
    return domain_halves(pool.domains(pool.test), seed=0)[0]


def load(pool, variant=""):
    """{sigma: (M,) flip rates} for this pool's members, or None if not computed."""
    path = cache_path(pool, variant)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    if c["ids"] != pool.ids:
        return None
    return {float(s): np.asarray(v) for s, v in c["flips"].items()}


def compute(pool, log=print, variant=""):
    from netkit import models as netkit_models
    dev = device()
    cand = candidate_rows(pool, variant)
    rows = np.sort(np.random.default_rng(SEED).choice(cand, min(POINTS, len(cand)), replace=False))
    X = target_images(pool, rows)
    gen = torch.Generator(device=dev)
    gen.manual_seed(SEED)
    flips = {s: [] for s in SIGMAS}
    t0 = time.time()
    for i, mid in enumerate(pool.ids):
        m = netkit_models.load_member(os.path.join(pool.root, "models", f"member_{mid:04d}.pt"), dev)
        clean = perturbed_argmax(m, X, dev, 0.0, gen)
        for s in SIGMAS:
            flips[s].append(float(np.mean([perturbed_argmax(m, X, dev, s, gen) != clean
                                           for _ in range(DRAWS)])))
        del m
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            log(f"  {pool.name}: {i + 1}/{pool.M} members  [{time.time() - t0:.0f}s]")
    out = {"ids": pool.ids, "rows": rows.tolist(), "draws": DRAWS,
           "flips": {str(s): v for s, v in flips.items()}}
    os.makedirs(os.path.dirname(cache_path(pool, variant)), exist_ok=True)
    with open(cache_path(pool, variant), "w", encoding="utf-8") as f:
        json.dump(out, f)
    return {s: np.asarray(v) for s, v in flips.items()}


def main(argv=None):
    from .run import POOLS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--inductive", action="store_true",
                    help="draw the inputs from the proposer half of the test split only")
    a = ap.parse_args(argv)
    variant = "inductive" if a.inductive else ""
    roots = list(a.pools)
    if a.all:
        roots = [os.path.join(POOLS, d) for d in sorted(os.listdir(POOLS))
                 if paper_pool(d)
                 and os.path.exists(os.path.join(POOLS, d, "predictions", "index.json"))]
    for root in roots:
        pool = Pool(root)
        if pool.config.get("mode", "build") != "build" or pool.M < pool.config.get("pool_size", 0):
            continue
        if variant and not pool.tuning:
            continue
        if load(pool, variant) is not None:
            print(f"{pool.name}: cached", flush=True)
            continue
        t0 = time.time()
        compute(pool, log=lambda s: print(s, flush=True), variant=variant)
        print(f"{pool.name}: done  [{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
