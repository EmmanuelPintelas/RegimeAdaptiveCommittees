"""Pack Camelyon17 into memory-mapped arrays, in the official WILDS splits.

The mirror stores the dataset as parquet with three splits, one of which merges
two official ones.  Every row carries its hospital, so the official protocol is
recovered exactly by filtering on ``center``:

    train      hospitals 0, 3, 4                 302,436
    id_val     hospitals 0, 3, 4 (held out)       33,560
    ood_val    hospital 1                         34,904
    ood_test   hospital 2                         85,054

Patches are uniform 96x96, so they are written as one uint8 array per split
rather than as loose files.  Forty pool members each stream the training split
repeatedly, and PNG decoding would otherwise dominate every epoch; a memmap gives
random access with no decode at all, at the cost of 12.6 GB on disk.

    python pack_camelyon.py --src ../data/camelyon17_hf --dst ../data/camelyon17
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

SIZE = 96
# split name -> (source parquet split, hospitals kept)
SPLITS = {
    "train":    ("train", None),
    "id_val":   ("validation", {0, 3, 4}),
    "ood_val":  ("validation", {1}),
    "ood_test": ("test", {2}),
}


def shards(src, split):
    return sorted(glob.glob(os.path.join(src, "data", f"{split}-*.parquet")))


def survey(src):
    """Read the small columns first so every array can be sized exactly."""
    meta = {}
    for name, (source, keep) in SPLITS.items():
        n = 0
        for f in shards(src, source):
            centers = pq.read_table(f, columns=["center"]).column("center").to_pylist()
            n += sum(1 for c in centers if keep is None or c in keep)
        meta[name] = n
        print(f"  {name:9} {n:>7} patches")
    return meta


def pack(src, dst, meta):
    os.makedirs(dst, exist_ok=True)
    arrays, labels, centers, offsets = {}, {}, {}, {}
    for name, n in meta.items():
        arrays[name] = np.lib.format.open_memmap(
            os.path.join(dst, f"{name}_x.npy"), mode="w+",
            dtype=np.uint8, shape=(n, SIZE, SIZE, 3))
        labels[name] = np.zeros(n, dtype=np.int64)
        centers[name] = np.zeros(n, dtype=np.int64)
        offsets[name] = 0

    t0 = time.time()
    done = 0
    total = sum(meta.values())
    for source in ("train", "validation", "test"):
        targets = [(n, keep) for n, (s, keep) in SPLITS.items() if s == source]
        for f in shards(src, source):
            table = pq.read_table(f, columns=["image", "label", "center"])
            imgs = table.column("image").to_pylist()
            labs = table.column("label").to_pylist()
            cens = table.column("center").to_pylist()
            for rec, lab, cen in zip(imgs, labs, cens):
                for name, keep in targets:
                    if keep is not None and cen not in keep:
                        continue
                    im = Image.open(io.BytesIO(rec["bytes"]))
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    if im.size != (SIZE, SIZE):
                        im = im.resize((SIZE, SIZE), Image.BILINEAR)
                    i = offsets[name]
                    arrays[name][i] = np.asarray(im, dtype=np.uint8)
                    labels[name][i] = lab
                    centers[name][i] = cen
                    offsets[name] = i + 1
                    break
                done += 1
                if done % 20000 == 0:
                    el = time.time() - t0
                    print(f"    {done}/{total}  {done/el:.0f} img/s  "
                          f"eta {(total-done)/max(done/el,1)/60:.0f} min", flush=True)
            del table, imgs

    for name in meta:
        arrays[name].flush()
        np.save(os.path.join(dst, f"{name}_y.npy"), labels[name])
        np.save(os.path.join(dst, f"{name}_center.npy"), centers[name])
        if offsets[name] != meta[name]:
            raise SystemExit(f"{name}: wrote {offsets[name]} of {meta[name]}")

    with open(os.path.join(dst, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"size": SIZE, "counts": meta,
                   "splits": {k: {"source": v[0],
                                  "hospitals": sorted(v[1]) if v[1] else "all"}
                              for k, v in SPLITS.items()}}, fh, indent=1)
    print(f"\npacked in {(time.time()-t0)/60:.1f} min")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="../data/camelyon17_hf")
    p.add_argument("--dst", default="../data/camelyon17")
    p.add_argument("--survey-only", action="store_true")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("surveying shards")
    meta = survey(a.src)
    expect = {"train": 302436, "id_val": 33560, "ood_val": 34904, "ood_test": 85054}
    if meta != expect:
        print(f"\ncounts differ from the official WILDS splits {expect}")
        return 1
    print("counts match the official WILDS splits exactly")
    if a.survey_only:
        return 0
    need = sum(meta.values()) * SIZE * SIZE * 3 / 1e9
    print(f"\npacking {sum(meta.values())} patches, about {need:.1f} GB")
    pack(a.src, a.dst, meta)
    print("\nnext (from the repository root):\n"
          "  python build_pool.py --config configs/pools/camelyon17.json --pool-seed 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
