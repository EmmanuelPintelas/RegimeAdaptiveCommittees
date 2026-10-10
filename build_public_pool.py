"""Pools of public ImageNet checkpoints on the imagenet_r200 benchmark (inference only).

Benchmark imagenet_r200: the 200 classes of ImageNet-R, logits restricted to them.
    tuning split  ImageNet-V2 (matched frequency) images of those classes, one domain
    test split    ImageNet-R (one domain per rendition type), and the ImageNet-Sketch and
                  ImageNet-A images of those classes (one domain each)

Pools:
    clip_soups  the 72 CLIP ViT-B/32 fine-tunes released with model soups (Wortsman et
                al. 2022), all from one linear-probe initialization; source-validation
                score = their published held-out ImageNet accuracy (ImageNet2p)
    timm_in1k   72 public timm models of distinct architectures trained on ImageNet-1k
                only, drawn with a fixed seed from timm's published results; source-
                validation score = their published ImageNet-1k top-1

Each pool is written in the layout build_pool.py uses (pool_config.json, manifest.jsonl,
predictions/<split>/{labels,domains,member_XXXX}.npy, predictions/index.json), with the
per-member weight statistics and the weight-noise cache the method's proposers read.
Every member is checked against its published accuracy on the full 1000-class
ImageNet-V2 (and, for CLIP, on ImageNet-R) before it is committed.

    python build_public_pool.py --pool clip_soups
    python build_public_pool.py --pool timm_in1k
    python build_public_pool.py --pool clip_soups --limit 2 --images 512   (quick test)

Resumable: members already committed are skipped.

Locations (environment variables):
    REGADCOM_PUBLIC   working folder (default <REGADCOM_DATA>/imagenet_r200); the four
                      archives go in <REGADCOM_PUBLIC>/data: imagenet-r.tar,
                      imagenetv2-matched-frequency.tar.gz, imagenet-a.tar and
                      ImageNet-Sketch.zip; checkpoints go in <REGADCOM_PUBLIC>/ckpt
    REGADCOM_STAGE    where the needed images are extracted, ideally a fast disk
                      (default <REGADCOM_PUBLIC>/stage)
    HF_HOME           the timm weight cache (default <REGADCOM_PUBLIC>/hf)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

PUBLIC = os.environ.get("REGADCOM_PUBLIC", os.path.join(paths.DATA, "imagenet_r200"))
os.environ.setdefault("HF_HOME", os.path.join(PUBLIC, "hf"))

import argparse  # noqa: E402
import csv  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from analysis.weight_noise import DRAWS, POINTS, SEED, SIGMAS, perturbed_argmax  # noqa: E402
from analysis.pools import CACHE  # noqa: E402
from poolkit.metrics import score_logits  # noqa: E402
from poolkit.weights import weight_stats  # noqa: E402

ARCHIVES = os.path.join(PUBLIC, "data")              # the downloaded archives
STAGE = os.environ.get("REGADCOM_STAGE", os.path.join(PUBLIC, "stage"))  # images read
STAGED = all(os.path.exists(os.path.join(STAGE, a + ".done")) for a in (
    "imagenet-r.tar", "imagenetv2-matched-frequency.tar.gz", "imagenet-a.tar",
    "ImageNet-Sketch.zip"))
DATA = STAGE if STAGED else ARCHIVES
BENCHMARK = "imagenet_r200"
M_TARGET = 72
TIMM_SEED = 0

SOUPS_URL = "https://github.com/mlfoundations/model-soups/releases/download/v0.0.2/model_{}.pt"
SOUPS_RAW = "https://raw.githubusercontent.com/mlfoundations/model-soups/main/"
TIMM_RESULTS = ("https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/"
                "results/")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# the ImageNet-R classes as indices into the 1000 ImageNet classes (model soups'
# datasets/imagenet_r.py); checked against the folders of imagenet-r
R200 = [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96,
        97, 99, 100, 105, 107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160,
        161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207, 208, 219, 231, 232, 234, 235,
        242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289,
        291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334,
        335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388,
        390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462,
        463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593,
        594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776,
        779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 889, 895, 907,
        928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963,
        965, 967, 980, 981, 983, 988]
IMG_EXT = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")
TOLERANCE = 0.01          # agreement with the published accuracy a member must reach


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --------------------------------------------------------------------------- #
# staging: the needed images, read sequentially from the archives onto a fast disk
# --------------------------------------------------------------------------- #
def stage(dest=STAGE):
    """Extract ImageNet-V2 (all), ImageNet-R (all), and the ImageNet-R classes of
    ImageNet-A and ImageNet-Sketch from their archives, in archive order (sequential reads
    of a slow drive), into the layout benchmark_splits reads.  Resumable."""
    import tarfile
    import zipfile
    os.makedirs(dest, exist_ok=True)

    def write(name, data):
        path = os.path.join(dest, *name.split("/"))
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)

    def from_tar(archive, keep):
        done = os.path.join(dest, os.path.basename(archive) + ".done")
        if os.path.exists(done):
            return
        n, t0 = 0, time.time()
        with tarfile.open(os.path.join(ARCHIVES, archive), "r|*") as tf:
            for m in tf:
                if m.isfile() and m.name.endswith(IMG_EXT) and keep(m.name):
                    write(m.name, tf.extractfile(m).read())
                    n += 1
        open(done, "w").close()
        log(f"  staged {n} images from {archive} [{time.time() - t0:.0f}s]")

    r_classes = set()
    from_tar("imagenet-r.tar", lambda name: r_classes.add(name.split("/")[1]) or True)
    if not r_classes:
        r_classes = set(os.listdir(os.path.join(dest, "imagenet-r")))
    from_tar("imagenetv2-matched-frequency.tar.gz", lambda name: True)
    from_tar("imagenet-a.tar", lambda name: name.split("/")[1] in r_classes)
    done = os.path.join(dest, "ImageNet-Sketch.zip.done")
    if not os.path.exists(done):
        t0 = time.time()
        with zipfile.ZipFile(os.path.join(ARCHIVES, "ImageNet-Sketch.zip")) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
            for wnid in {i.filename.split("/")[1] for i in infos if i.filename.count("/") >= 2}:
                os.makedirs(os.path.join(dest, "sketch", wnid), exist_ok=True)
            keep = sorted((i for i in infos if i.filename.endswith(IMG_EXT)
                           and i.filename.split("/")[1] in r_classes),
                          key=lambda i: i.header_offset)
            for i in keep:
                write(i.filename, z.read(i))
        open(done, "w").close()
        log(f"  staged {len(keep)} images from ImageNet-Sketch.zip [{time.time() - t0:.0f}s]")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def find_dir(*candidates):
    for c in candidates:
        p = os.path.join(DATA, c)
        if os.path.isdir(p):
            return p
    raise SystemExit(f"none of {candidates} found under {DATA}")


def images_in(folder):
    return sorted(f for f in os.listdir(folder) if f.endswith(IMG_EXT))


def benchmark_splits():
    """{split: (paths, labels, domains)} and the domain names, deterministic order."""
    sketch = find_dir("sketch", os.path.join("ImageNet-Sketch", "sketch"),
                      os.path.join("imagenet-sketch", "sketch"))
    wnids = sorted(d for d in os.listdir(sketch) if os.path.isdir(os.path.join(sketch, d)))
    if len(wnids) != 1000:
        raise SystemExit(f"expected 1000 ImageNet classes in {sketch}, found {len(wnids)}")
    rdir = find_dir("imagenet-r")
    rw = sorted(d for d in os.listdir(rdir) if os.path.isdir(os.path.join(rdir, d)))
    if [wnids.index(w) for w in rw] != R200:
        raise SystemExit("the ImageNet-R folders do not match the 200-class list")
    pos = {w: i for i, w in enumerate(rw)}
    domains = []

    def dom(name):
        if name not in domains:
            domains.append(name)
        return domains.index(name)

    v2 = find_dir("imagenetv2-matched-frequency-format-val")
    tun, full = ([], [], []), ([], [], [])
    for c in range(1000):
        folder = os.path.join(v2, str(c))
        for f in images_in(folder):
            p = os.path.join(folder, f)
            full[0].append(p), full[1].append(c), full[2].append(0)
            if wnids[c] in pos:
                tun[0].append(p), tun[1].append(pos[wnids[c]]), tun[2].append(0)
    test = ([], [], [])
    for w in rw:
        for f in images_in(os.path.join(rdir, w)):
            test[0].append(os.path.join(rdir, w, f)), test[1].append(pos[w])
            test[2].append(dom("r_" + f.split("_")[0]))
    for w in rw:
        for f in images_in(os.path.join(sketch, w)):
            test[0].append(os.path.join(sketch, w, f)), test[1].append(pos[w])
            test[2].append(dom("sketch"))
    adir = find_dir("imagenet-a")
    for w in sorted(os.listdir(adir)):
        if w in pos and os.path.isdir(os.path.join(adir, w)):
            for f in images_in(os.path.join(adir, w)):
                test[0].append(os.path.join(adir, w, f)), test[1].append(pos[w])
                test[2].append(dom("imagenet_a"))
    as_np = lambda t: (t[0], np.asarray(t[1], np.int64), np.asarray(t[2], np.int64))  # noqa
    tun_doms = ["imagenet_v2"]
    return ({"ood_val": as_np(tun), "ood_test": as_np(test), "v2_full": as_np(full)},
            {"ood_val": tun_doms, "ood_test": domains, "v2_full": tun_doms})


class Files(Dataset):
    def __init__(self, files, tf):
        self.files, self.tf = files, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            x = self.tf(im.convert("RGB"))
        return (x * 255).round().to(torch.uint8)


def pre_transform(size, crop_pct, interpolation):
    """Resize and center crop, no normalization: exact as uint8."""
    from torchvision import transforms as T
    interp = {"bicubic": T.InterpolationMode.BICUBIC, "bilinear": T.InterpolationMode.BILINEAR,
              "nearest": T.InterpolationMode.NEAREST}[interpolation]
    resize = int(np.floor(size / crop_pct)) if crop_pct else size      # as timm does
    return T.Compose([T.Resize(resize, interpolation=interp), T.CenterCrop(size), T.ToTensor()])


CACHE_DIR = os.path.join(os.path.dirname(STAGE.rstrip("/\\")), "cache")  # fast disk


def image_cache(key, files, tf, workers):
    """uint8 (N, 3, H, W) memmap of the preprocessed images, built once per preprocessing."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}_{len(files)}.u8")
    probe = Files(files[:1], tf)[0]
    shape = (len(files),) + tuple(probe.shape)
    if os.path.exists(path + ".done"):
        return np.memmap(path, np.uint8, "r", shape=shape)
    mm = np.memmap(path, np.uint8, "w+", shape=shape)
    dl = DataLoader(Files(files, tf), batch_size=128, num_workers=workers)
    i, t0 = 0, time.time()
    for x in dl:
        mm[i:i + len(x)] = x.numpy()
        i += len(x)
    mm.flush()
    open(path + ".done", "w").close()
    log(f"  cached {len(files)} images as {key} [{time.time() - t0:.0f}s]")
    return np.memmap(path, np.uint8, "r", shape=shape)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
class Normalized(torch.nn.Module):
    """uint8 images in, logits over the selected classes out."""

    def __init__(self, net, mean, std, classes=None):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255)
        self.register_buffer("cls", torch.as_tensor(classes) if classes is not None
                             else torch.zeros(0, dtype=torch.long))

    def forward(self, x):
        out = self.net((x.float() - self.mean) / self.std)
        return out[:, self.cls] if len(self.cls) else out


class ClipClassifier(torch.nn.Module):
    """Model soups' ModelWrapper: normalized CLIP image embedding, then a linear head."""

    def __init__(self, visual, head):
        super().__init__()
        self.visual, self.head = visual, head

    def forward(self, x):
        f = self.visual(x)
        return self.head(f / f.norm(dim=-1, keepdim=True))


def fetch(url, dest):
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def fetch_text(url, cache_name):
    path = os.path.join(PUBLIC, "meta", cache_name)
    fetch(url, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def clip_member(i, dev):
    import timm
    from timm.models.vision_transformer import _convert_openai_clip
    sd = torch.load(fetch(SOUPS_URL.format(i), os.path.join(PUBLIC, "ckpt", f"model_{i}.pt")),
                    map_location="cpu", weights_only=False)
    visual = timm.create_model("vit_base_patch32_clip_quickgelu_224", pretrained=False,
                               num_classes=512)
    vis = {k[len("model."):]: v.float() for k, v in sd.items() if k.startswith("model.visual.")}
    visual.load_state_dict(_convert_openai_clip(vis, visual, prefix="visual."), strict=True)
    head = torch.nn.Linear(512, 1000)
    head.weight.data = sd["classification_head.weight"].float()
    head.bias.data = sd["classification_head.bias"].float()
    net = ClipClassifier(visual, head)
    return net.to(dev).eval(), ("clip", 224, None, "bicubic", CLIP_MEAN, CLIP_STD)


def timm_member(m, dev):
    """The model, preprocessed as in timm's published results for it (size, crop, resize)."""
    import timm
    net = timm.create_model(m["name"], pretrained=True)
    cfg = timm.data.resolve_model_data_config(net)
    if cfg.get("crop_mode", "center") != "center":
        raise RuntimeError(f"crop mode {cfg.get('crop_mode')}")
    r = m["recipe"]
    size, crop, interp = r["img_size"], r["crop_pct"], r["interpolation"]
    key = f"s{size}_c{crop:.4f}_{interp}"
    return net.to(dev).eval(), (key, size, crop, interp, tuple(cfg["mean"]), tuple(cfg["std"]))


# --------------------------------------------------------------------------- #
# member lists
# --------------------------------------------------------------------------- #
def clip_members():
    res = {json.loads(l)["model_name"]: json.loads(l) for l in
           fetch_text(SOUPS_RAW + "individual_model_results.jsonl",
                      "soups_individual_model_results.jsonl").splitlines() if l.strip()}
    hp = json.loads(fetch_text(SOUPS_RAW + "hparam_info.json", "soups_hparam_info.json"))
    out = []
    for i in range(M_TARGET):
        r = res[f"model_{i}"]
        out.append({"name": f"model_{i}", "arch": "vit_base_patch32_clip", "source_val":
                    r["ImageNet2p"], "published": r, "recipe": hp.get(f"model_{i}", {}),
                    "checkpoint": SOUPS_URL.format(i)})
    return out


def read_csv(text):
    return list(csv.DictReader(io.StringIO(text)))


IN1K_ONLY_PREFIXES = {"", "tv", "tv2", "a1", "a1h", "a2", "a3", "ra", "ra2", "ra3", "ra4", "rsb",
                      "gluon", "tf", "fb", "ms", "mx", "bt", "c", "d", "e", "sail", "nav",
                      "ap", "aa", "ra_ch", "tf_ap", "tf_aa", "miil", "snap", "apple", "hf"}
EXCLUDE = ("clip", "eva", "beit", "dino", "mae", "siglip", "laion", "openai", "ssl", "swsl",
           "ig", "_ns", "jft", "distill", "dist", "in21k", "in22k", "in12k", "m38m", "sw_")


def timm_candidates():
    """ImageNet-1k-only timm models with published results, one tag per architecture, in a
    fixed random order (seed TIMM_SEED); the pool takes the first M_TARGET that load."""
    import timm
    avail = set(timm.list_models(pretrained=True))

    def rows_by_model(text):
        out = {}
        for r in read_csv(text):
            out.setdefault(r["model"], []).append(r)
        return out
    top1_all = rows_by_model(fetch_text(TIMM_RESULTS + "results-imagenet.csv",
                                        "timm_results_imagenet.csv"))
    v2_all = rows_by_model(fetch_text(TIMM_RESULTS + "results-imagenetv2-matched-frequency.csv",
                                      "timm_results_v2.csv"))
    top1, v2, by_arch = {}, {}, {}
    for name in sorted(top1_all):
        if name not in avail or name not in v2_all or "." not in name:
            continue
        arch, tag = name.split(".", 1)
        if not tag.endswith("in1k") or any(x in name for x in EXCLUDE):
            continue
        prefix = tag[:-len("in1k")].rstrip("_")
        if prefix not in IN1K_ONLY_PREFIXES:
            continue
        size = int(timm.models.get_pretrained_cfg(name).input_size[-1])
        r1 = [r for r in top1_all[name] if int(r["img_size"]) == size]
        r2 = [r for r in v2_all[name] if int(r["img_size"]) == size]
        if not r1 or not r2:
            continue
        top1[name], v2[name] = r1[0], r2[0]
        if float(r1[0]["param_count"]) > 100 or size > 288:
            continue
        by_arch.setdefault(arch, name)
    names = sorted(by_arch.values())
    order = np.random.default_rng(TIMM_SEED).permutation(len(names))
    return [{"name": names[j], "arch": names[j].split(".")[0],
             "source_val": float(top1[names[j]]["top1"]) / 100,
             "published": {"ImageNet": float(top1[names[j]]["top1"]) / 100,
                           "ImageNetV2": float(v2[names[j]]["top1"]) / 100},
             "recipe": {"timm": names[j], "img_size": int(v2[names[j]]["img_size"]),
                        "crop_pct": float(v2[names[j]]["crop_pct"]),
                        "interpolation": v2[names[j]]["interpolation"],
                        "param_count_m": float(top1[names[j]]["param_count"])},
             "checkpoint": f"timm:{names[j]}"} for j in order]


# --------------------------------------------------------------------------- #
# the pool
# --------------------------------------------------------------------------- #
@torch.no_grad()
def logits_of(model, mm, dev, bs=256):
    out = []
    for i in range(0, len(mm), bs):
        x = torch.from_numpy(np.array(mm[i:i + bs])).to(dev, non_blocking=True)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def scores(logits, labels, K):
    s = score_logits(logits, labels, K)
    return {k: float(v) for k, v in s.items() if isinstance(v, (int, float, np.floating))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", choices=("clip_soups", "timm_in1k"))
    ap.add_argument("--stage", action="store_true",
                    help="copy the benchmark's images from the archives to the fast disk")
    ap.add_argument("--out", help="default <BUILD_POOLS>/public_<pool>")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="members (quick test)")
    ap.add_argument("--images", type=int, help="images per split (quick test)")
    args = ap.parse_args(argv)
    if args.stage:
        stage()
        return 0
    if args.pool is None:
        ap.error("--pool is required")
    if DATA != STAGE:
        log(f"note: reading images from {DATA}; run --stage first to copy them to {STAGE}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.abspath(args.out or os.path.join(
        paths.BUILD_POOLS, f"{'test_' if args.limit or args.images else ''}public_{args.pool}"))
    pred = os.path.join(out, "predictions")

    splits, dom_names = benchmark_splits()
    if args.images:
        rng = np.random.default_rng(0)
        for s, (f, y, d) in list(splits.items()):
            keep = np.sort(rng.choice(len(f), min(args.images, len(f)), replace=False))
            splits[s] = ([f[i] for i in keep], y[keep], d[keep])
    K = 200
    for s in ("ood_val", "ood_test"):
        os.makedirs(os.path.join(pred, s), exist_ok=True)
        np.save(os.path.join(pred, s, "labels.npy"), splits[s][1])
        np.save(os.path.join(pred, s, "domains.npy"), splits[s][2])
    files = splits["ood_val"][0] + splits["ood_test"][0] + splits["v2_full"][0]
    n_v, n_t = len(splits["ood_val"][0]), len(splits["ood_test"][0])
    sl = {"ood_val": slice(0, n_v), "ood_test": slice(n_v, n_v + n_t),
          "v2_full": slice(n_v + n_t, len(files))}
    noise_rows = np.sort(np.random.default_rng(SEED).choice(n_t, min(POINTS, n_t), replace=False))

    members = clip_members() if args.pool == "clip_soups" else timm_candidates()
    target = args.limit or M_TARGET
    cfg_path = os.path.join(out, "pool_config.json")
    if not os.path.exists(cfg_path):
        config = {"benchmark": BENCHMARK, "mode": "public", "pool": args.pool,
                  "image_size": 224, "pool_size": target, "num_classes": K,
                  "candidates": [m["name"] for m in members],
                  "splits": {"ood_val": "ImageNet-V2 matched frequency, ImageNet-R classes",
                             "ood_test": "ImageNet-R, ImageNet-Sketch and ImageNet-A, "
                                         "ImageNet-R classes"},
                  "domains": {s: dom_names[s] for s in ("ood_val", "ood_test")}}
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"schema": "public_pool/1", "created_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config": config}, f, indent=1)
    man_path = os.path.join(out, "manifest.jsonl")
    rows = [json.loads(l) for l in open(man_path, encoding="utf-8")] \
        if os.path.exists(man_path) else []
    done = {r["name"] for r in rows}
    failed_path = os.path.join(out, "failed.jsonl")
    failed = {json.loads(l)["name"] for l in open(failed_path, encoding="utf-8")} \
        if os.path.exists(failed_path) else set()
    noise_path = os.path.join(CACHE, os.path.basename(out), "weight_noise.json")
    noise = {"ids": [], "rows": noise_rows.tolist(), "draws": DRAWS,
             "flips": {str(s): [] for s in SIGMAS}}
    if os.path.exists(noise_path):                     # keep the committed members' entries
        old = json.load(open(noise_path, encoding="utf-8"))
        pos_of = {i: j for j, i in enumerate(old["ids"])}
        missing = [r["member_id"] for r in rows if r["member_id"] not in pos_of]
        if missing or old["rows"] != noise["rows"]:
            raise SystemExit(f"weight-noise record does not cover members {missing}")
        for r in rows:
            noise["ids"].append(r["member_id"])
            for s in SIGMAS:
                noise["flips"][str(s)].append(old["flips"][str(s)][pos_of[r["member_id"]]])
    caches = {}
    log(f"{args.pool}: {len(rows)}/{target} committed; tuning {n_v}, test {n_t} images "
        f"({len(dom_names['ood_test'])} domains), V2 check {len(splits['v2_full'][0])}")

    def key_of(m):
        if args.pool == "clip_soups":
            return "clip"
        r = m["recipe"]
        return f"s{r['img_size']}_c{r['crop_pct']:.4f}_{r['interpolation']}"

    def drop_cache(key):
        mm = caches.pop(key, None)
        if mm is not None:
            path = mm.filename
            del mm
            for p in (path, path + ".done", path + f".noise{len(noise_rows)}",
                      path + f".noise{len(noise_rows)}.done"):
                if os.path.exists(p):
                    os.remove(p)

    def plan():
        """The next members to try: the first candidates in the fixed order that are
        neither committed nor failed, as many as are missing, grouped by preprocessing."""
        todo = [m for m in members if m["name"] not in done and m["name"] not in failed]
        todo = todo[:target - len(rows)]
        keys = list(dict.fromkeys(key_of(m) for m in todo))
        return [m for k in keys for m in todo if key_of(m) == k]

    def process(m):
        """Run one candidate; its manifest row if it is committed, None if it fails."""
        t0 = time.time()
        try:
            net, (key, size, crop, interp, mean, std) = (
                clip_member(int(m["name"].split("_")[1]), dev) if args.pool == "clip_soups"
                else timm_member(m, dev))
        except Exception as exc:                       # a checkpoint that will not load
            with open(failed_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"name": m["name"], "error": repr(exc)[:300]}) + "\n")
            log(f"  {m['name']}: skipped ({exc!r:.120})")
            return None
        if key not in caches:
            tf = pre_transform(size, crop or 1.0, interp)
            caches[key] = image_cache(f"{key}_{args.images or 'all'}", files, tf, args.workers)
        mm = caches[key]
        full = Normalized(net, mean, std).to(dev).eval()
        lg_full = logits_of(full, mm[sl["v2_full"]], dev)
        v2_acc = float((lg_full.argmax(1) == splits["v2_full"][1]).mean())
        model = Normalized(net, mean, std, R200).to(dev).eval()
        lg = {s: logits_of(model, mm[sl[s]], dev) for s in ("ood_val", "ood_test")}
        r_mask = np.char.startswith(np.asarray(dom_names["ood_test"])[splits["ood_test"][2]], "r_")
        r_acc = float((lg["ood_test"][r_mask].argmax(1) == splits["ood_test"][1][r_mask]).mean())
        verify = {"v2_acc": v2_acc, "v2_published": m["published"]["ImageNetV2"]}
        if "ImageNetR" in m["published"]:
            verify.update({"r_acc": r_acc, "r_published": m["published"]["ImageNetR"]})
        if not args.images:
            bad = [k for k in ("v2", "r") if f"{k}_published" in verify
                   and abs(verify[f"{k}_acc"] - verify[f"{k}_published"]) > TOLERANCE]
            if bad:
                with open(failed_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"name": m["name"], "error": "accuracy mismatch",
                                        "verify": verify}) + "\n")
                log(f"  {m['name']}: does not reproduce its published accuracy {verify}")
                return None
        mid = len(rows)
        for s in ("ood_val", "ood_test"):
            np.save(os.path.join(pred, s, f"member_{mid:04d}.npy"), lg[s].astype(np.float16))
        npath = mm.filename + f".noise{len(noise_rows)}"
        if not os.path.exists(npath + ".done"):
            sub = np.memmap(npath, np.uint8, "w+", shape=(len(noise_rows),) + mm.shape[1:])
            sub[:] = mm[sl["ood_test"]][noise_rows]
            sub.flush()
            del sub
            open(npath + ".done", "w").close()
        X = torch.from_numpy(np.array(np.memmap(npath, np.uint8, "r",
                                                shape=(len(noise_rows),) + mm.shape[1:])))
        gen = torch.Generator(device=dev)
        gen.manual_seed(SEED + mid)
        clean = perturbed_argmax(model, X, dev, 0.0, gen)
        for s in SIGMAS:
            noise["flips"][str(s)].append(float(np.mean(
                [perturbed_argmax(model, X, dev, s, gen) != clean for _ in range(DRAWS)])))
        noise["ids"].append(mid)
        ws = weight_stats({k: v for k, v in net.state_dict().items()}) \
            if args.pool == "clip_soups" else {}
        row = {"member_id": mid, "name": m["name"], "status": "accepted", "reasons": [],
               "mode": "public", "arch": m["arch"], "recipe": m["recipe"],
               "metrics": {"source_val": {k: m["source_val"] for k in
                                          ("acc", "balanced_acc", "macro_f1", "gm")},
                           **{s: scores(lg[s], splits[s][1], K) for s in lg}},
               "weight_stats": ws, "pred_dtype": "float16",
               "public": {"checkpoint": m["checkpoint"], "published": m["published"],
                          "verify": verify},
               "seconds": {"total": time.time() - t0}}
        os.makedirs(os.path.dirname(noise_path), exist_ok=True)
        with open(noise_path, "w", encoding="utf-8") as f:     # before the manifest row
            json.dump(noise, f)
        with open(man_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        with open(os.path.join(pred, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"schema": "poolkit.predictions/1", "benchmark": BENCHMARK,
                       "num_classes": K, "image_size": 224,
                       "content": "pre-softmax logits over the 200 ImageNet-R classes",
                       "splits": {"ood_val": {"role": "tuning", "n": n_v,
                                              "labels": "ood_val/labels.npy",
                                              "domains": "ood_val/domains.npy"},
                                  "ood_test": {"role": "test", "n": n_t,
                                               "labels": "ood_test/labels.npy",
                                               "domains": "ood_test/domains.npy"}},
                       "accepted": [r["member_id"] for r in rows] + [mid]}, f, indent=1)
        log(f"  [{mid + 1}/{target}] {m['name']}: V2 {v2_acc:.4f} "
            f"(published {m['published']['ImageNetV2']:.4f})"
            + (f", R {r_acc:.4f} (published {m['published']['ImageNetR']:.4f})"
               if "ImageNetR" in m["published"] else "")
            + f", test macro-F1 {row['metrics']['ood_test']['macro_f1']:.4f}"
            f"  [{time.time() - t0:.0f}s]")
        del net, full, model
        torch.cuda.empty_cache()
        return row

    while len(rows) < target:
        work = plan()
        if not work:
            break
        for n_m, m in enumerate(work):
            row = process(m)
            if row is not None:
                rows.append(row)
                done.add(m["name"])
            else:
                failed.add(m["name"])
            if args.pool != "clip_soups" and all(key_of(x) != key_of(m) for x in work[n_m + 1:]):
                drop_cache(key_of(m))
    if len(rows) >= target:                            # free the fast disk
        for key in list(caches):
            drop_cache(key)
    log(f"{args.pool}: {len(rows)}/{target} members")
    return 0 if len(rows) >= target else 1


if __name__ == "__main__":
    sys.exit(main())
