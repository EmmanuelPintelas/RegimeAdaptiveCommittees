"""Pack the Kather colorectal cohorts into memory-mapped arrays.

Two archives from Zenodo record 1214456, 224x224 patches at 0.5 MPP over nine
tissue classes:

    NCT-CRC-HE-100K.zip   100,000 patches, 86 patients, NCT Heidelberg and UMM
    CRC-VAL-HE-7K.zip       7,180 patches, 50 patients, no overlap with the above

    python pack_kather.py --src ../data/kather_raw --dst ../data/kather

The 100K is the source and is split here, once, into a training part and an
in-distribution validation part, stratified by class under a fixed seed and
recorded in the manifest.  The 7K is the target domain and is never split.
There is no published in-distribution validation split for this data, so one has
to be made; making it here rather than at load time means every run sees the
same one and the pool stays comparable across runs.

Read the caveat before reading any result from this benchmark.  Both released
archives are Macenko colour-normalised.  The staining variation between labs is
therefore already largely removed, which is precisely the variation the stain
family models and the mechanism the paper leans on for Camelyon17.  What remains
between the two cohorts is a patient and preparation shift, which is real but
weaker.  A NONORM variant exists for the 100K only; pairing it against the
normalised 7K would put a preprocessing difference where the domain shift is
supposed to be, and would measure the pipeline rather than the data, so this
script refuses that combination rather than offering it.

The archives are read directly.  Extracting 11.7 GB of TIFFs to write a 3 GB
array and then delete them again is churn for nothing.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile

import numpy as np
from PIL import Image

# The nine classes, in the fixed alphabetical order both archives use.  Pinned
# rather than discovered, so that a missing class directory is an error instead
# of a silent relabelling of everything after it.
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
COHORTS = {"NCT-CRC-HE-100K": 0, "CRC-VAL-HE-7K": 1}
EXPECT = {"NCT-CRC-HE-100K": 100000, "CRC-VAL-HE-7K": 7180}


def members(zf, cohort):
    """(class index, member name) for every image in the archive, sorted.

    Sorted by name so the packing order is reproducible, which is what makes the
    recorded train/id_val split meaningful across machines.
    """
    out = []
    index = {c: i for i, c in enumerate(CLASSES)}
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if not name.lower().endswith((".tif", ".tiff", ".png", ".jpg")):
            continue
        parts = name.replace("\\", "/").split("/")
        cls = next((p for p in parts if p in index), None)
        if cls is None:
            raise SystemExit(f"{cohort}: {name} is under no known class directory")
        out.append((index[cls], name))
    out.sort(key=lambda t: t[1])
    return out


def survey(src):
    found = {}
    for cohort in COHORTS:
        path = os.path.join(src, f"{cohort}.zip")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
        if os.path.exists(os.path.join(src, "NCT-CRC-HE-100K-NONORM.zip")) \
                and cohort == "NCT-CRC-HE-100K":
            print("  note: a NONORM archive is present and is being ignored; "
                  "see the caveat in this script's docstring")
        with zipfile.ZipFile(path) as zf:
            items = members(zf, cohort)
        found[cohort] = items
        per_class = np.bincount([c for c, _ in items], minlength=len(CLASSES))
        print(f"  {cohort:18} {len(items):>7} patches  "
              f"per class {per_class.tolist()}")
        if len(items) != EXPECT[cohort]:
            raise SystemExit(f"{cohort}: found {len(items)}, "
                             f"the published count is {EXPECT[cohort]}")
    return found


def split_source(labels, val_frac, seed):
    """Stratified index split of the source cohort, by class."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    for c in range(len(CLASSES)):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * val_frac))
        val.append(idx[:k])
        train.append(idx[k:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(val))


def write_split(zf, items, order, name, dst, size, cohort_id):
    n = len(order)
    x = np.lib.format.open_memmap(os.path.join(dst, f"{name}_x.npy"), mode="w+",
                                  dtype=np.uint8, shape=(n, size, size, 3))
    y = np.zeros(n, dtype=np.int64)
    dom = np.full(n, cohort_id, dtype=np.int64)
    t0 = time.time()
    for j, i in enumerate(order):
        cls, member = items[int(i)]
        with zf.open(member) as fh:
            img = Image.open(io.BytesIO(fh.read()))
            if img.mode != "RGB":
                img = img.convert("RGB")
            if img.size != (size, size):
                img = img.resize((size, size), Image.BILINEAR)
            x[j] = np.asarray(img, dtype=np.uint8)
        y[j] = cls
        if (j + 1) % 20000 == 0:
            el = time.time() - t0
            print(f"    {name}: {j + 1}/{n}  {(j + 1) / el:.0f} img/s", flush=True)
    x.flush()
    np.save(os.path.join(dst, f"{name}_y.npy"), y)
    np.save(os.path.join(dst, f"{name}_domain.npy"), dom)
    print(f"  {name:9} {n:>7} patches, {n * size * size * 3 / 1e9:.1f} GB, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="../data/kather_raw")
    p.add_argument("--dst", default="../data/kather")
    p.add_argument("--size", type=int, default=128,
                   help="side of the packed square; the archives are 224")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--survey-only", action="store_true")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print(f"surveying {os.path.abspath(a.src)}")
    found = survey(a.src)

    src_items = found["NCT-CRC-HE-100K"]
    src_labels = np.array([c for c, _ in src_items])
    tr, va = split_source(src_labels, a.val_frac, a.seed)
    counts = {"train": len(tr), "id_val": len(va),
              "ood_test": len(found["CRC-VAL-HE-7K"])}
    total = sum(counts.values())
    for k, v in counts.items():
        print(f"  {k:9} {v:>7}")
    gb = total * a.size * a.size * 3 / 1e9
    print(f"  packing {total} patches at {a.size}px = {gb:.1f} GB")
    if a.survey_only:
        return 0

    free = __import__("shutil").disk_usage(os.path.abspath("../data")).free / 1e9
    if free < gb * 1.1:
        print(f"  only {free:.0f} GB free; need about {gb * 1.1:.0f} GB")
        return 1

    os.makedirs(a.dst, exist_ok=True)
    t0 = time.time()
    written = {}
    with zipfile.ZipFile(os.path.join(a.src, "NCT-CRC-HE-100K.zip")) as zf:
        written["train"] = write_split(zf, src_items, tr, "train", a.dst, a.size,
                                       COHORTS["NCT-CRC-HE-100K"])
        written["id_val"] = write_split(zf, src_items, va, "id_val", a.dst, a.size,
                                        COHORTS["NCT-CRC-HE-100K"])
    with zipfile.ZipFile(os.path.join(a.src, "CRC-VAL-HE-7K.zip")) as zf:
        items = found["CRC-VAL-HE-7K"]
        written["ood_test"] = write_split(zf, items, np.arange(len(items)),
                                          "ood_test", a.dst, a.size,
                                          COHORTS["CRC-VAL-HE-7K"])
    if written != counts:
        raise SystemExit(f"wrote {written}, expected {counts}")

    manifest = {
        "benchmark": "kather",
        "size": a.size,
        "num_classes": len(CLASSES),
        "classes": CLASSES,
        "counts": counts,
        "source_train": "train",
        "source_val": "id_val",
        "targets": {"ood_test": "ood_test_crc_val_7k"},
        "domain_field": "cohort",
        "domains": COHORTS,
        "val_frac": a.val_frac,
        "seed": a.seed,
        "note": "source is NCT-CRC-HE-100K (86 patients, NCT Heidelberg and UMM "
                "Mannheim), target is CRC-VAL-HE-7K (50 non-overlapping "
                "patients, every filename tagged TCGA); both archives are "
                "Macenko colour-normalised, so the staining component of the "
                "shift is largely removed",
        "source_archive": "zenodo.org/records/1214456",
    }
    with open(os.path.join(a.dst, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    print(f"\npacked {total} patches in {(time.time() - t0) / 60:.1f} min -> "
          f"{os.path.abspath(a.dst)}")
    print(f"\nnext:\n  python train_pool2.py --benchmark kather --data {a.dst} "
          f"--pool-size 40 --image-size 96 --setups random_family "
          f"--out pools/kather_families")
    return 0


if __name__ == "__main__":
    sys.exit(main())
