"""Pack a WILDS image task into memory-mapped arrays, in its official splits.

The same argument as pack_camelyon.py, for the tasks whose archives arrive as
loose image files rather than as parquet.  Forty pool members each stream the
training split repeatedly, and JPEG or PNG decoding would otherwise dominate
every epoch; a uint8 memmap gives random access with no decode at all.

    python pack_wilds.py --task rxrx1   --src ../data/wilds --dst ../data/rxrx1
    python pack_wilds.py --task iwildcam --src ../data/wilds --dst ../data/iwildcam

The splits are not reimplemented here.  They are read off the wilds package's
own dataset object, which owns the official protocol, so a change in the
protocol cannot silently leave this script packing something else.  What this
script decides is only the resolution and the on-disk layout.

Two of the four split roles need a word.

The source validation split has to be in-distribution: it stands in for the
labelled data a practitioner really has, and the whole claim of the method is
that it is used without ever seeing the target.  iWildCam publishes an id_val
for exactly this.  RxRx1 does not, and its only in-distribution split is
id_test, the second imaging site of the same 33 experiments; that is what is
packed as id_val here.  Falling back to the OOD val instead -- which is what a
naive `id_val or val` would do -- would put target-domain data into checkpoint
selection and quietly invalidate every number computed downstream.

Resolution is a choice and not a property of the data.  Camelyon17 is packed at
its native 96 and loses nothing.  These two are packed at 128 by default, which
is above the 96 the pools train at, so a later run at 128 needs no repack, and
far below what the images carry.  For iWildCam in particular that is a real
loss: WILDS trains it at 448, the animals are often small in frame, and a pool
trained at 96 is solving a harder problem than the published one.  The number is
recorded in the manifest so no run can be confused about what it read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

# task -> (packed split name, wilds split name) plus the metadata field that
# names the domain.  The domain is stored alongside the labels so that a per-
# domain breakdown never needs the raw archive again.
TASKS = {
    "rxrx1": {
        "splits": [("train", "train"), ("id_val", "id_test"),
                   ("ood_val", "val"), ("ood_test", "test")],
        "domain": "experiment",
        "note": "domains are 51 imaging experiments; id_val is site 2 of the "
                "33 training experiments, since rxrx1 publishes no id_val",
    },
    "iwildcam": {
        "splits": [("train", "train"), ("id_val", "id_val"),
                   ("ood_val", "val"), ("ood_test", "test")],
        "domain": "location",
        "note": "domains are camera-trap locations; train and id_val share "
                "them, ood_val and ood_test hold out disjoint ones",
    },
}


def open_dataset(task, src):
    """The wilds dataset object, working around one incompatibility.

    RxRx1's loader in the wilds package builds its split array with
    ``df.dataset.apply(...).values`` and then writes the id_test split into it.
    Under pandas 3, copy-on-write hands back a read-only view, so that write
    dies with "assignment destination is read-only" and the task cannot be
    constructed at all.  Returning a writable copy from ``.values`` for the
    duration of the call is the smallest fix that leaves site-packages alone, and
    it is semantically a no-op: every caller gets the same numbers, in an array
    it is allowed to modify.  The property is restored in a finally block so
    nothing outside this call sees the patched pandas.
    """
    import pandas as pd
    from wilds import get_dataset

    original = pd.Series.values
    pd.Series.values = property(
        lambda self: np.array(original.fget(self), copy=True))
    try:
        return get_dataset(dataset=task, root_dir=src, download=False)
    finally:
        pd.Series.values = original


def survey(ds, task):
    """Index arrays per packed split, read off the wilds split array."""
    split_array = np.asarray(ds.split_array)
    out = {}
    for packed, official in TASKS[task]["splits"]:
        code = ds.split_dict.get(official)
        if code is None:
            raise SystemExit(f"{task} has no split {official!r}; "
                             f"it publishes {sorted(ds.split_dict)}")
        idx = np.where(split_array == code)[0]
        if len(idx) == 0:
            raise SystemExit(f"{task}: split {official!r} is empty")
        out[packed] = idx
    return out


def domain_values(ds, task, idx):
    """The domain id of each example, as an integer, or zeros if unavailable."""
    field = TASKS[task]["domain"]
    fields = list(ds.metadata_fields)
    if field not in fields:
        print(f"  note: no {field!r} field in metadata; domains stored as 0")
        return np.zeros(len(idx), dtype=np.int64)
    col = fields.index(field)
    return np.asarray(ds.metadata_array[idx, col]).astype(np.int64)


def _load_one(ds, i, size):
    """One image, decoded as small as the format allows and resized to size."""
    img = ds.get_input(int(i))
    # draft() lets libjpeg decode straight to a reduced scale, which is most of
    # the cost on iWildCam's multi-megapixel frames.  It is a no-op for PNG.
    try:
        img.draft("RGB", (size, size))
    except Exception:
        pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def pack_split(ds, task, name, idx, dst, size, workers):
    n = len(idx)
    x = np.lib.format.open_memmap(os.path.join(dst, f"{name}_x.npy"), mode="w+",
                                  dtype=np.uint8, shape=(n, size, size, 3))
    y = np.asarray(ds.y_array[idx]).astype(np.int64)
    dom = domain_values(ds, task, idx)

    t0, done = time.time(), 0

    def work(j):
        x[j] = _load_one(ds, idx[j], size)

    # Threads rather than processes: PIL releases the GIL inside the decoder, and
    # each thread writes to its own row of the memmap, so nothing is shared that
    # would need locking or a round trip through pickle.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(work, range(n), chunksize=64):
            done += 1
            if done % 20000 == 0:
                el = time.time() - t0
                print(f"    {name}: {done}/{n}  {done / el:.0f} img/s  "
                      f"eta {(n - done) / max(done / el, 1) / 60:.0f} min",
                      flush=True)
    x.flush()
    np.save(os.path.join(dst, f"{name}_y.npy"), y)
    np.save(os.path.join(dst, f"{name}_domain.npy"), dom)
    print(f"  {name:9} {n:>7} images, {len(np.unique(dom))} domains, "
          f"{n * size * size * 3 / 1e9:.1f} GB, {(time.time() - t0) / 60:.1f} min",
          flush=True)
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--src", default="../data/wilds",
                   help="root holding <task>_v<version>, as fetch_wilds.py leaves it")
    p.add_argument("--dst", default=None, help="default ../data/<task>")
    p.add_argument("--size", type=int, default=128,
                   help="side of the packed square; 128 leaves headroom over "
                        "the 96 the pools train at")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--survey-only", action="store_true")
    a = p.parse_args(argv)
    dst = a.dst or os.path.join("../data", a.task)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print(f"opening {a.task} from {os.path.abspath(a.src)}")
    ds = open_dataset(a.task, a.src)
    print(f"  {len(ds)} examples, {ds.n_classes} classes, "
          f"splits {sorted(ds.split_dict)}")

    idx = survey(ds, a.task)
    counts = {k: int(len(v)) for k, v in idx.items()}
    total = sum(counts.values())
    for k, v in counts.items():
        print(f"  {k:9} {v:>7}")
    gb = total * a.size * a.size * 3 / 1e9
    print(f"  packing {total} images at {a.size}px = {gb:.1f} GB")
    if a.survey_only:
        return 0

    free = __import__("shutil").disk_usage(os.path.abspath("../data")).free / 1e9
    if free < gb * 1.1:
        print(f"  only {free:.0f} GB free; need about {gb * 1.1:.0f} GB")
        return 1

    os.makedirs(dst, exist_ok=True)
    t0 = time.time()
    written = {}
    for name in idx:
        written[name] = pack_split(ds, a.task, name, idx[name], dst, a.size,
                                   a.workers)
    if written != counts:
        raise SystemExit(f"wrote {written}, expected {counts}")

    manifest = {
        "benchmark": a.task,
        "size": a.size,
        "num_classes": int(ds.n_classes),
        "counts": counts,
        "source_train": "train",
        "source_val": "id_val",
        "targets": {"ood_val": "ood_val", "ood_test": "ood_test"},
        "domain_field": TASKS[a.task]["domain"],
        "split_map": {k: v for k, v in TASKS[a.task]["splits"]},
        "note": TASKS[a.task]["note"],
        "source": "official WILDS splits, read from the wilds package",
    }
    with open(os.path.join(dst, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    print(f"\npacked {total} images in {(time.time() - t0) / 60:.1f} min -> "
          f"{os.path.abspath(dst)}")
    print(f"\nnext:\n  python train_pool2.py --benchmark {a.task} --data {dst} "
          f"--pool-size 40 --image-size 96 --setups random_family "
          f"--out pools/{a.task}_families")
    return 0


if __name__ == "__main__":
    sys.exit(main())
