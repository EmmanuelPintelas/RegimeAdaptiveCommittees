"""Benchmark loading, with the domain structure the method needs.

Every benchmark is presented through one interface:

    Benchmark(name, num_classes, source_train, source_val, targets)

where ``targets`` maps a held-out domain name to a labelled dataset.  Pool
members are trained on ``source_train`` and selected on ``source_val``; the
target domains are never touched during pool construction, and at method time
their labels are used only to score, never to choose.

Two families are supported.  DomainBed-style sets (PACS, VLCS, OfficeHome) are
directory trees of the form ``root/<domain>/<class>/*.jpg`` and need no
dependency beyond torchvision, so they are read natively.  WILDS tasks are read
through the ``wilds`` package if it is installed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder

# The four real-world tasks of GeNeX, each with a distribution-shifted split
# produced by ratio-constrained Jensen-Shannon-divergence-guided clustering, and
# a random split of the same data as the control.
GENEX_TASKS = ["SKIN_CANCER", "CHEST_XRAY", "DEEPFAKE", "PLANT_DISEASE"]
GENEX_VARIANTS = {"drift": "DISTRIBUTIONAL_DRIFTED", "random": "RANDOM_SAMPLING"}

DOMAINBED = {
    "PACS": ["art_painting", "cartoon", "photo", "sketch"],
    "VLCS": ["Caltech101", "LabelMe", "SUN09", "VOC2007"],
    "OfficeHome": ["Art", "Clipart", "Product", "Real_World"],
}
WILDS_TASKS = ["camelyon17", "iwildcam", "fmow", "rxrx1"]


@dataclass
class Benchmark:
    name: str
    num_classes: int
    source_train: Dataset
    source_val: Dataset
    targets: Dict[str, Dataset]
    held_out: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def describe(self):
        tgt = ", ".join(f"{k} n={len(v)}" for k, v in self.targets.items())
        return (f"{self.name}"
                + (f" (held out: {self.held_out})" if self.held_out else "")
                + f"\n  classes      {self.num_classes}"
                + f"\n  source train {len(self.source_train)}"
                + f"\n  source val   {len(self.source_val)}"
                + f"\n  targets      {tgt}")


class Transformed(Dataset):
    """Wrap a dataset so the transform can be swapped per pool member.

    Pool members disagree partly because they see different augmentations, so
    the same underlying images must be readable through several pipelines
    without paying for a second copy on disk.
    """

    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, target = self.base[i]
        return (self.transform(img) if self.transform is not None else img), target


class _RawImageFolder(ImageFolder):
    """ImageFolder that returns PIL images, leaving transforms to Transformed."""

    def __init__(self, root, classes=None):
        super().__init__(root, transform=None)
        if classes is not None and list(self.classes) != list(classes):
            raise ValueError(f"class mismatch in {root}: {self.classes} vs {classes}")

    @property
    def targets_array(self):
        return np.asarray(self.targets)


def _stratified_split(n_targets, frac, seed):
    """Split indices per class so the validation set keeps the class balance."""
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(n_targets):
        idx = np.flatnonzero(n_targets == c)
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * frac)))
        va.extend(idx[:cut].tolist())
        tr.extend(idx[cut:].tolist())
    return sorted(tr), sorted(va)


def load_domainbed(name, root, held_out, val_frac=0.2, seed=0):
    """Leave-one-domain-out over a DomainBed directory tree."""
    if name not in DOMAINBED:
        raise ValueError(f"unknown DomainBed set {name}; expected one of {list(DOMAINBED)}")
    domains = DOMAINBED[name]
    if held_out not in domains:
        raise ValueError(f"{held_out} is not a domain of {name}: {domains}")

    base = os.path.join(root, name)
    folders = {d: _RawImageFolder(os.path.join(base, d)) for d in domains}
    classes = folders[domains[0]].classes
    for d, f in folders.items():
        if list(f.classes) != list(classes):
            raise ValueError(f"domain {d} has classes {f.classes}, expected {classes}")

    source_parts, val_parts = [], []
    for d in domains:
        if d == held_out:
            continue
        f = folders[d]
        tr, va = _stratified_split(f.targets_array, val_frac, seed)
        source_parts.append(Subset(f, tr))
        val_parts.append(Subset(f, va))

    from torch.utils.data import ConcatDataset
    return Benchmark(
        name=f"{name}",
        num_classes=len(classes),
        source_train=ConcatDataset(source_parts),
        source_val=ConcatDataset(val_parts),
        targets={held_out: folders[held_out]},
        held_out=held_out,
        meta={"family": "domainbed", "domains": domains, "classes": classes},
    )


def load_wilds(task, root, val_frac=None, seed=0):
    """Official WILDS splits: train / id_val / (val, test) as target domains."""
    try:
        from wilds import get_dataset
    except ImportError as e:
        raise ImportError(
            "the wilds package is required for WILDS tasks; install it with "
            "`pip install wilds`") from e
    if task not in WILDS_TASKS:
        raise ValueError(f"unknown WILDS task {task}; expected one of {WILDS_TASKS}")

    ds = get_dataset(dataset=task, root_dir=root, download=False)
    n_classes = int(ds.n_classes)

    def subset(split):
        try:
            return ds.get_subset(split, transform=None)
        except (KeyError, ValueError):
            return None

    train = subset("train")
    # The source validation split must be in-distribution.  Falling back to the
    # OOD val when a task publishes no id_val -- rxrx1 does not -- would select
    # checkpoints on target-domain data and then also score on it, which is the
    # one thing the whole design forbids; and the identity test that was meant
    # to catch it cannot, since get_subset builds a fresh object every call.  So
    # the fallback is the other in-distribution split, and the OOD splits are
    # named explicitly rather than inferred by elimination.
    id_name = next((s for s in ("id_val", "id_test") if subset(s) is not None), None)
    if id_name is None:
        raise RuntimeError(
            f"{task}: no in-distribution validation split (id_val or id_test); "
            "hold one out of train before using this task")
    id_val = subset(id_name)
    targets = {f"ood_{s}": subset(s) for s in ("val", "test") if subset(s) is not None}
    if not targets:
        raise RuntimeError(f"{task}: no out-of-distribution split found")

    return Benchmark(
        name=task,
        num_classes=n_classes,
        source_train=_Wrap(train),
        source_val=_Wrap(id_val),
        targets={k: _Wrap(v) for k, v in targets.items()},
        held_out=None,
        meta={"family": "wilds"},
    )


class _Wrap(Dataset):
    """WILDS subsets yield (x, y, metadata); drop the metadata."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        out = self.base[i]
        return (out[0], int(out[1])) if len(out) >= 2 else out


def load_genex(task, root, variant="drift"):
    """One of the four GeNeX tasks, in its shifted or its random split.

    The directory tree is ``<root>/<TASK>/<VARIANT>/{train,val,test}/<class>``.
    ``train`` and ``val`` are the source; ``test`` is the target, and under the
    drifted variant it is separated from the source by construction rather than
    by chance, which is the setting in which relying on validation is known to
    fail.  The random variant holds the data fixed and removes only the shift,
    so the pair isolates what the shift is responsible for.
    """
    if task not in GENEX_TASKS:
        raise ValueError(f"unknown GeNeX task {task}; expected one of {GENEX_TASKS}")
    if variant not in GENEX_VARIANTS:
        raise ValueError(f"unknown variant {variant}; expected one of "
                         f"{list(GENEX_VARIANTS)}")
    base = os.path.join(root, task, GENEX_VARIANTS[variant])
    if not os.path.isdir(base):
        raise FileNotFoundError(f"{base} not found")

    def folder(split):
        d = os.path.join(base, split)
        return _RawImageFolder(d) if os.path.isdir(d) else None

    train, val = folder("train"), folder("val")
    if train is None or val is None:
        raise FileNotFoundError(f"{base}: expected train/ and val/ subdirectories")

    # SKIN_CANCER ships both the full target set and a 10% probe used for quick
    # checks during its construction.  Where both are present, `_test` is the
    # real one; the probe is kept addressable but is not the target.
    candidates = {name: folder(name) for name in ("_test", "test")}
    candidates = {k: v for k, v in candidates.items() if v is not None}
    if not candidates:
        raise FileNotFoundError(f"{base}: no test/ or _test/ subdirectory")
    main = "_test" if "_test" in candidates else "test"
    targets = {"target": candidates[main]}
    for k, v in candidates.items():
        if k != main:
            targets["probe_10pct"] = v

    classes = train.classes
    for name, f in list(targets.items()) + [("val", val)]:
        if list(f.classes) != list(classes):
            raise ValueError(f"{name} has classes {f.classes}, expected {classes}")

    return Benchmark(
        name=f"{task} ({variant})",
        num_classes=len(classes),
        source_train=train,
        source_val=val,
        targets=targets,
        held_out=None,
        meta={"family": "genex", "variant": variant, "classes": classes,
              "target_split": main},
    )


class MemmapImages(Dataset):
    """Uniform-size patches held as one memory-mapped uint8 array.

    Camelyon17 is 456,000 patches of 96x96, streamed repeatedly by every pool
    member.  Held as loose PNGs the decode cost dominates each epoch; held as a
    memmap the reads are random-access and free, and the operating system keeps
    the hot pages resident.  The array is opened lazily per worker so that a
    forked DataLoader does not inherit a shared file handle.
    """

    def __init__(self, x_path, y_path, domain_path=None):
        self.x_path, self.y_path = x_path, y_path
        self.y = np.load(y_path)
        self.domains = np.load(domain_path) if domain_path else None
        # Camelyon17's domain is its hospital and was packed under that name;
        # the alias keeps anything reading .centers working.
        self.centers = self.domains
        self._x = None

    def __len__(self):
        return len(self.y)

    @property
    def x(self):
        if self._x is None:
            self._x = np.load(self.x_path, mmap_mode="r")
        return self._x

    def __getitem__(self, i):
        from PIL import Image
        return Image.fromarray(np.asarray(self.x[i])), int(self.y[i])

    @property
    def classes(self):
        return [str(c) for c in sorted(set(self.y.tolist()))]


def load_camelyon17(root):
    """Camelyon17 in the official WILDS splits, from packed arrays.

    Source is hospitals 0, 3 and 4; validation is their held-out patches; the
    two target domains are hospital 1 and hospital 2, neither seen in training.
    """
    man = os.path.join(root, "manifest.json")
    if not os.path.exists(man):
        raise FileNotFoundError(
            f"{root} has no manifest.json; run pack_camelyon.py first")
    import json
    meta = json.load(open(man, encoding="utf-8"))

    def split(name):
        return MemmapImages(os.path.join(root, f"{name}_x.npy"),
                            os.path.join(root, f"{name}_y.npy"),
                            os.path.join(root, f"{name}_center.npy"))

    return Benchmark(
        name="camelyon17",
        num_classes=2,
        source_train=split("train"),
        source_val=split("id_val"),
        targets={"ood_val_hospital1": split("ood_val"),
                 "ood_test_hospital2": split("ood_test")},
        held_out=None,
        meta={"family": "camelyon17", "counts": meta["counts"],
              "splits": meta["splits"]},
    )


def load_packed(root, name=None):
    """Any benchmark written by pack_wilds.py or pack_kather.py.

    The packers record everything the loader needs in manifest.json -- which
    split is the source, which is the in-distribution validation, which are the
    targets, and at what resolution the arrays were written -- so one loader
    serves every packed dataset and adding another needs no code here.

    The domain array packed beside each split is carried on the dataset object
    rather than dropped, because a per-domain breakdown of a target split is the
    natural diagnostic and it should not require the raw archive again.
    """
    import json

    man_path = os.path.join(root, "manifest.json")
    if not os.path.exists(man_path):
        raise FileNotFoundError(f"{root} has no manifest.json; run the packer first")
    with open(man_path, encoding="utf-8") as fh:
        man = json.load(fh)

    def split(key):
        dom = os.path.join(root, f"{key}_domain.npy")
        return MemmapImages(os.path.join(root, f"{key}_x.npy"),
                            os.path.join(root, f"{key}_y.npy"),
                            dom if os.path.exists(dom) else None)

    targets = man.get("targets") or {k: k for k in man["counts"]
                                     if k.startswith("ood")}
    return Benchmark(
        name=name or man.get("benchmark", os.path.basename(root.rstrip("/\\"))),
        num_classes=int(man["num_classes"]),
        source_train=split(man.get("source_train", "train")),
        source_val=split(man.get("source_val", "id_val")),
        targets={label: split(key) for key, label in targets.items()},
        held_out=None,
        meta={"family": "packed", "size": man["size"], "counts": man["counts"],
              "domain_field": man.get("domain_field"), "note": man.get("note")},
    )


# Benchmarks that live on disk as packed arrays, each with its own packer.
PACKED_TASKS = ["rxrx1", "iwildcam", "kather"]


def load(spec, root, val_frac=0.2, seed=0):
    """Dispatch on a spec.

    ``PACS:sketch``            leave-one-domain-out over DomainBed
    ``genex:SKIN_CANCER``      GeNeX task, distribution-shifted split
    ``genex:SKIN_CANCER:random``   the same task, random split (control)
    ``camelyon17``             a WILDS task
    ``rxrx1`` ``iwildcam`` ``kather``   a packed benchmark, see pack_wilds.py
    """
    if spec == "camelyon17" and os.path.exists(os.path.join(root, "manifest.json")):
        return load_camelyon17(root)
    if spec in PACKED_TASKS and os.path.exists(os.path.join(root, "manifest.json")):
        return load_packed(root, spec)
    if spec.lower().startswith("genex:"):
        parts = spec.split(":")
        task = parts[1]
        variant = parts[2] if len(parts) > 2 else "drift"
        return load_genex(task, root, variant)
    if ":" in spec:
        name, held_out = spec.split(":", 1)
        return load_domainbed(name, root, held_out, val_frac, seed)
    if spec in DOMAINBED:
        raise ValueError(f"{spec} needs a held-out domain, e.g. "
                         f"'{spec}:{DOMAINBED[spec][-1]}'")
    return load_wilds(spec, root, val_frac, seed)
