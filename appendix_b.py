"""Appendix B of the paper: alternatives to selection that did not hold up.

    python appendix_b.py ranks
    python appendix_b.py soups   --pool pools/kather_seed_1
    python appendix_b.py soups   --pool pools/rxrx1_seed_1
    python appendix_b.py noise   --pool pools/camelyon17_seed_1
    python appendix_b.py noise   --pool pools/camelyon17_convnext_atto_seed_2
    python appendix_b.py distill --pool pools/rxrx1_seed_1 --init member
    python appendix_b.py distill --pool pools/rxrx1_seed_1 --init fresh
    python appendix_b.py distill --pool pools/rxrx1_seed_1 --init member --temperature 3 --epochs 24
    python appendix_b.py distill --pool pools/rxrx1_seed_1 --students 3

ranks    rank-averaging the spectral exponent with confidence, against each alone
         (Spearman with the members' test scores); no GPU, no image
soups    weight-space averages of a single-backbone pool, raw and with batch-norm
         statistics recalibrated on unlabeled target images, for the whole pool and
         for three clusters of the members' prediction vectors
noise    prediction flips under per-tensor scaled Gaussian weight noise, as a
         selection signal against confidence
distill  the full-pool committee distilled into one student, or into k students on
         bootstrap shards with one student on their union as the control; this one
         trains networks

Target labels are read only to score.  The teacher, the prediction vectors and the member
scores come from the stored predictions.  Results go to results/appendix_b/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import paths
from analysis.committees import Scorer
from analysis.pools import Pool, device
from analysis.run import POOLS
from analysis.weight_noise import perturbed_argmax
from netkit import data as D
from netkit import models as netkit_models
from netkit.augment import eval_transform
from netkit.data import Transformed

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "appendix_b")


# --------------------------------------------------------------------------- #
# shared
# --------------------------------------------------------------------------- #
def log(msg):
    print(msg, flush=True)


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    log(f"wrote {path}")


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, float))).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def partial_spearman(a, b, c):
    """Spearman of a and b with c held constant, on ranks."""
    R = np.array([np.argsort(np.argsort(np.asarray(v, float))).astype(float) for v in (a, b, c)])
    C = np.corrcoef(R)
    den = np.sqrt((1 - C[0, 2] ** 2) * (1 - C[1, 2] ** 2))
    return float((C[0, 1] - C[0, 2] * C[1, 2]) / den) if den else float("nan")


def bench_of(pool):
    return D.load(pool.benchmark, os.path.join(paths.DATA, pool.benchmark))


def member_path(pool, mid):
    return os.path.join(pool.root, "models", f"member_{mid:04d}.pt")


def stored_probs(pool, split, rows=None):
    """(M, n, K) float32 softmax of the stored logits, on the CPU."""
    out = []
    for mid in pool.ids:
        lg = np.load(pool.logit_path(split, mid))
        if rows is not None:
            lg = lg[rows]
        out.append(torch.softmax(torch.from_numpy(lg.astype(np.float32)), dim=1).numpy())
    return np.stack(out)


def member_scores(pool, split):
    y = pool.labels(split)
    sc = Scorer(y, pool.K, torch.device("cpu"))
    return np.array([sc(torch.from_numpy(np.load(pool.logit_path(split, m)).argmax(1)), pool.metric)
                     for m in pool.ids])


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            z = model(x.to(dev, non_blocking=True))
        out.append(torch.softmax(z.float(), dim=1).cpu())
    return torch.cat(out).numpy()


def score(pool, probs, y):
    return Scorer(y, pool.K, torch.device("cpu"))(torch.from_numpy(probs.argmax(1)), pool.metric)


# --------------------------------------------------------------------------- #
# ranks
# --------------------------------------------------------------------------- #
def cmd_ranks(a):
    rows = []
    for d in sorted(os.listdir(POOLS)):
        root = os.path.join(POOLS, d)
        if not os.path.exists(os.path.join(root, "predictions", "index.json")):
            continue
        pool = Pool(root)
        if pool.config.get("mode", "build") != "build" or not pool.single_arch():
            continue
        alpha = pool.weight_stat("alpha")
        if not np.isfinite(alpha).all():
            continue
        ms = member_scores(pool, pool.test)
        conf = pool.compact(pool.test, log=log)["conf"].astype(np.float32).mean(axis=1)
        r_conf = np.argsort(np.argsort(conf))
        row = {"pool": pool.name, "benchmark": pool.benchmark, "confidence": spearman(conf, ms)}
        for tag, s in (("alpha_low", -alpha), ("alpha_high", alpha)):
            comb = r_conf + np.argsort(np.argsort(s))
            row[tag] = spearman(s, ms)
            row[f"conf+{tag}"] = spearman(comb, ms)
        rows.append(row)
        log(f"{pool.name}: " + ", ".join(f"{k} {v:+.3f}" for k, v in row.items()
                                         if isinstance(v, float)))
    save("ranks.json", {"rows": rows})


# --------------------------------------------------------------------------- #
# soups
# --------------------------------------------------------------------------- #
def kmeans(x, k, seed=0, iters=100, restarts=20):
    best, best_inertia = None, np.inf
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        c = x[rng.choice(len(x), k, replace=False)].copy()
        lab = None
        for _ in range(iters):
            new = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(2), axis=1)
            if lab is not None and (new == lab).all():
                break
            lab = new
            for j in range(k):
                if (lab == j).any():
                    c[j] = x[lab == j].mean(0)
        inertia = float(((x - c[lab]) ** 2).sum())
        if inertia < best_inertia:
            best, best_inertia = lab, inertia
    return best


def average_weights(paths_):
    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths_]
    if len({c["arch"] for c in ckpts}) != 1:
        raise SystemExit("weight soups need a single-backbone pool")
    sd = {}
    for key, ref in ckpts[0]["state_dict"].items():
        if ref.is_floating_point():
            acc = torch.zeros_like(ref, dtype=torch.float64)
            for c in ckpts:
                acc += c["state_dict"][key].to(torch.float64)
            sd[key] = (acc / len(ckpts)).to(ref.dtype)
        else:
            sd[key] = ref.clone()
    m = netkit_models.PoolMember(ckpts[0]["arch"], ckpts[0]["num_classes"], pretrained=False,
                                image_size=ckpts[0].get("image_size", 224))
    m.load_state_dict(sd)
    return m


@torch.no_grad()
def recalibrate_bn(model, loader, dev, batches):
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for i, (x, _) in enumerate(loader):
        model(x.to(dev, non_blocking=True))
        if i + 1 >= batches:
            break
    model.eval()
    return model


def cmd_soups(a):
    dev = device()
    pool = Pool(a.pool)
    if not pool.single_arch():
        raise SystemExit("weight soups need a single-backbone pool")
    split = pool.test
    y = pool.labels(split)
    bench = bench_of(pool)
    tf = eval_transform(pool.config["image_size"])
    tgt = bench.targets[split]
    loader = DataLoader(Transformed(tgt, tf), batch_size=256, shuffle=False,
                        num_workers=a.workers, pin_memory=dev.type == "cuda")
    bn_loader = DataLoader(Transformed(tgt, tf), batch_size=256, shuffle=True,
                           num_workers=a.workers, generator=torch.Generator().manual_seed(0))
    log(f"{pool.name}: {pool.M} members, {split} n={len(y)}, metric {pool.metric}")

    # prediction vectors and prediction-space references, from the stored predictions
    sig_rows = np.sort(np.random.default_rng(0).choice(len(y), min(512, len(y)), replace=False))
    sig = stored_probs(pool, split, sig_rows).reshape(pool.M, -1).astype(np.float64)
    xc = sig - sig.mean(0)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    space = u[:, :2] * s[:2]
    assign = kmeans(space, a.k)
    clusters = {int(j): [pool.ids[i] for i in np.flatnonzero(assign == j)] for j in np.unique(assign)}
    total = None
    for mid in pool.ids:
        p = torch.softmax(torch.from_numpy(np.load(pool.logit_path(split, mid)).astype(np.float32)), 1)
        total = p if total is None else total + p
    res = {"pool": pool.name, "split": split, "metric": pool.metric,
           "prediction_average": score(pool, total.numpy(), y),
           "best_member": float(member_scores(pool, split).max()),
           "clusters": {str(j): len(v) for j, v in clusters.items()}}
    log(f"  prediction average {res['prediction_average']:.4f}, best member "
        f"{res['best_member']:.4f}; clusters {res['clusters']}")

    def run_soup(tag, ids):
        t0 = time.time()
        m = average_weights([member_path(pool, i) for i in ids]).to(dev)
        raw = score(pool, predict(m, loader, dev), y)
        recalibrate_bn(m, bn_loader, dev, a.bn_batches)
        bn = score(pool, predict(m, loader, dev), y)
        log(f"  {tag:10s} ({len(ids):3d} members): raw {raw:.4f}, batch-norm recalibrated "
            f"{bn:.4f}  [{time.time() - t0:.0f}s]")
        del m
        torch.cuda.empty_cache()
        return {"members": len(ids), "raw": raw, "bn": bn}

    res["whole_pool"] = run_soup("whole pool", pool.ids)
    res["cluster_soups"] = {str(j): run_soup(f"cluster {j}", ids) for j, ids in clusters.items()}
    save(f"soups_{pool.name}.json", res)


def cmd_greedy_soup(a):
    """Greedy soup (Wortsman et al. 2022): ingredients in order of their score on the tuning
    split, each kept if the soup's tuning score does not fall.  A soup averages every
    floating-point tensor of its ingredients, batch-norm statistics included, so a soup of one
    is that member exactly.  The final soup is scored on the test split as averaged and after
    recalibrating its batch-norm statistics on the unlabeled test inputs."""
    dev = device()
    pool = Pool(a.pool)
    if not pool.single_arch() or not pool.tuning:
        raise SystemExit("greedy soup needs a single-backbone pool with a tuning split")
    bench = bench_of(pool)
    tf = eval_transform(pool.config["image_size"])

    def loader(split, shuffle=False):
        return DataLoader(Transformed(bench.targets[split], tf), batch_size=256, shuffle=shuffle,
                          num_workers=a.workers, pin_memory=dev.type == "cuda",
                          generator=torch.Generator().manual_seed(0) if shuffle else None)
    v_loader, t_loader = loader(pool.tuning), loader(pool.test)
    t_bn = loader(pool.test, shuffle=True)
    y_v, y_t = pool.labels(pool.tuning), pool.labels(pool.test)
    order = [pool.ids[i] for i in np.argsort(-member_scores(pool, pool.tuning), kind="stable")]
    ckpt0 = torch.load(member_path(pool, order[0]), map_location="cpu", weights_only=False)
    floats = [k for k, v in ckpt0["state_dict"].items() if v.is_floating_point()]
    total = {k: ckpt0["state_dict"][k].to(torch.float64).clone() for k in floats}

    def soup_model(tot, n, bn_loader=None):
        sd = {k: (tot[k] / n).to(ckpt0["state_dict"][k].dtype) if k in tot
              else ckpt0["state_dict"][k].clone() for k in ckpt0["state_dict"]}
        m = netkit_models.PoolMember(ckpt0["arch"], ckpt0["num_classes"], pretrained=False,
                                    image_size=ckpt0.get("image_size", 224))
        m.load_state_dict(sd)
        m = m.to(dev).eval()
        if bn_loader is not None:
            recalibrate_bn(m, bn_loader, dev, a.bn_batches)
        return m

    kept = [order[0]]
    best = score(pool, predict(soup_model(total, 1), v_loader, dev), y_v)
    first_test = score(pool, predict(soup_model(total, 1), t_loader, dev), y_t)
    log(f"{pool.name}: start from member {order[0]}, tuning {best:.4f}, test {first_test:.4f}")
    for mid in order[1:]:
        t0 = time.time()
        sd = torch.load(member_path(pool, mid), map_location="cpu", weights_only=False)["state_dict"]
        trial = {k: total[k] + sd[k].to(torch.float64) for k in floats}
        m = soup_model(trial, len(kept) + 1)
        s = score(pool, predict(m, v_loader, dev), y_v)
        del m
        torch.cuda.empty_cache()
        if s >= best:
            total, best = trial, s
            kept.append(mid)
        log(f"  member {mid}: tuning {s:.4f} -> {'kept' if s >= best and kept[-1] == mid else 'skipped'}"
            f" ({len(kept)} ingredients)  [{time.time() - t0:.0f}s]")
    test_avg = score(pool, predict(soup_model(total, len(kept)), t_loader, dev), y_t)
    test_bn = score(pool, predict(soup_model(total, len(kept), t_bn), t_loader, dev), y_t)
    res = {"pool": pool.name, "ingredients": kept, "tuning": best, "test": test_avg,
           "test_bn_recalibrated": test_bn, "best_member": order[0],
           "best_member_test": first_test}
    log(f"  greedy soup: {len(kept)} ingredients, tuning {best:.4f}, test {test_avg:.4f} "
        f"(BN recalibrated on test inputs {test_bn:.4f}); best member test {first_test:.4f}")
    save(f"greedy_soup_{pool.name}.json", res)


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #
def cmd_noise(a):
    dev = device()
    pool = Pool(a.pool)
    split = pool.test
    y = pool.labels(split)
    sigmas = [float(v) for v in a.sigmas.split(",")]
    bench = bench_of(pool)
    rows = np.sort(np.random.default_rng(a.seed).choice(len(y), min(a.points, len(y)), replace=False))
    dl = DataLoader(Transformed(Subset(bench.targets[split], rows.tolist()),
                                eval_transform(pool.config["image_size"])),
                    batch_size=256, shuffle=False, num_workers=a.workers)
    X = torch.cat([x for x, _ in dl])
    y_sub = y[rows]
    ms = member_scores(pool, split)
    conf = pool.compact(split, log=log)["conf"].astype(np.float32).mean(axis=1)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)
    flips = {s: np.zeros(pool.M) for s in sigmas}
    on_right = {s: np.zeros(pool.M) for s in sigmas}
    on_wrong = {s: np.zeros(pool.M) for s in sigmas}
    t0 = time.time()
    for i, mid in enumerate(pool.ids):
        m = netkit_models.load_member(member_path(pool, mid), dev)
        clean = perturbed_argmax(m, X, dev, 0.0, gen)
        right = clean == y_sub
        for s in sigmas:
            f = np.mean([perturbed_argmax(m, X, dev, s, gen) != clean for _ in range(a.draws)], axis=0)
            flips[s][i] = f.mean()
            on_right[s][i] = f[right].mean() if right.any() else np.nan
            on_wrong[s][i] = f[~right].mean() if (~right).any() else np.nan
        del m
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            log(f"  {i + 1}/{pool.M} members  [{time.time() - t0:.0f}s]")
    res = {"pool": pool.name, "split": split, "points": int(len(rows)), "draws": a.draws,
           "spearman_confidence": spearman(conf, ms), "by_sigma": {}}
    for s in sigmas:
        ratio = np.nanmean(on_right[s]) / max(np.nanmean(on_wrong[s]), 1e-12)
        res["by_sigma"][str(s)] = {
            "mean_flip_rate": float(flips[s].mean()),
            "flip_right_over_wrong": float(ratio),
            "spearman_low_flips": spearman(-flips[s], ms),
            "spearman_high_flips": spearman(flips[s], ms),
            "partial_given_confidence": partial_spearman(flips[s], ms, conf)}
        log(f"  sigma {s}: flip rate {flips[s].mean():.4f}, right/wrong {ratio:.2f}, "
            f"Spearman(score, -flips) {spearman(-flips[s], ms):+.3f} vs confidence "
            f"{res['spearman_confidence']:+.3f}; partial given confidence "
            f"{res['by_sigma'][str(s)]['partial_given_confidence']:+.3f}")
    save(f"noise_{pool.name}.json", res)


# --------------------------------------------------------------------------- #
# distill
# --------------------------------------------------------------------------- #
class WithTeacher(torch.utils.data.Dataset):
    """Target images paired with the teacher's output, never with a label."""

    def __init__(self, base, tf, probs, index):
        self.base, self.tf, self.probs, self.index = base, tf, probs, index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = int(self.index[i])
        img, _ = self.base[j]
        return self.tf(img), torch.from_numpy(self.probs[j])


def warm_up(student, bench, tf, dev, a):
    wl = DataLoader(Transformed(bench.source_train, tf), batch_size=a.batch_size, shuffle=True,
                    num_workers=a.workers, drop_last=True)
    opt = torch.optim.Adam(student.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    for ep in range(a.warmup_epochs):
        student.train()
        for b, (x, y) in enumerate(wl):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                loss = F.cross_entropy(student(x.to(dev)), y.to(dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if a.smoke and b >= 3:
                break
        log(f"    warm-up epoch {ep + 1} on {len(bench.source_train)} source images done")


def distil(student, tgt, tf, teach, index, dev, a, tag):
    loader = DataLoader(WithTeacher(tgt, tf, teach, index), batch_size=a.batch_size, shuffle=True,
                        num_workers=a.workers, drop_last=True)
    opt = torch.optim.Adam(student.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    T = a.temperature
    for ep in range(a.epochs):
        student.train()
        tot, nb = 0.0, 0
        for b, (x, q) in enumerate(loader):
            q = q.to(dev)
            if T != 1.0:
                q = q.clamp_min(1e-12).pow(1.0 / T)
                q = q / q.sum(1, keepdim=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                logp = F.log_softmax(student(x.to(dev)).float() / T, dim=1)
                loss = -(q * logp).sum(1).mean() * (T * T)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss)
            nb += 1
            if a.smoke and b >= 3:
                break
        sched.step()
        log(f"    {tag} epoch {ep + 1}/{a.epochs}: distillation loss {tot / max(nb, 1):.4f}")
    return student


def cmd_distill(a):
    dev = device()
    pool = Pool(a.pool)
    split = pool.test
    y = pool.labels(split)
    bench = bench_of(pool)
    tgt = bench.targets[split]
    tf = eval_transform(pool.config["image_size"])
    teach = None
    for mid in pool.ids:                      # running sum: RxRx1 cannot hold all members
        p = torch.softmax(torch.from_numpy(np.load(pool.logit_path(split, mid)).astype(np.float32)), 1)
        teach = p if teach is None else teach + p
    teach = (teach / pool.M).numpy()
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(y))
    n_ho = int(round(len(y) * a.heldout_frac))
    ho, tr = np.sort(perm[:n_ho]), np.sort(perm[n_ho:])
    if a.smoke:
        ho, tr = ho[:512], tr[:1024]
    ho_loader = DataLoader(Transformed(Subset(tgt, ho.tolist()), tf), batch_size=256,
                           shuffle=False, num_workers=a.workers)
    t_ho = score(pool, teach[ho], y[ho])
    log(f"{pool.name}: distil on {len(tr)} target images, hold out {len(ho)}; teacher on the "
        f"held-out part {t_ho:.4f} ({pool.M} members)")
    src = pool.source_val_score()

    def new_student(init_id):
        if a.init == "member" or init_id is not None:
            mid = init_id if init_id is not None else pool.ids[int(np.argmax(src))]
            return netkit_models.load_member(member_path(pool, mid), dev), f"member {mid}"
        arch = pool.archs[0]
        st = netkit_models.build(arch, pool.K, pretrained=True, device=dev,
                                image_size=pool.config["image_size"])
        warm_up(st, bench, tf, dev, a)
        return st, f"fresh {arch}"

    def evaluate_student(st):
        pr = predict(st, ho_loader, dev)
        return pr, score(pool, pr, y[ho]), float(np.sqrt(((pr - teach[ho]) ** 2).sum(1).mean()))

    res = {"pool": pool.name, "split": split, "heldout": int(len(ho)), "distil": int(len(tr)),
           "teacher_heldout": t_ho, "init": a.init, "temperature": a.temperature,
           "epochs": a.epochs, "students": a.students, "smoke": bool(a.smoke)}
    if a.students <= 1:
        st, how = new_student(None)
        _, s0, d0 = evaluate_student(st)
        log(f"  student ({how}) before distillation {s0:.4f}, Delta {d0:.4f}")
        distil(st, tgt, tf, teach, tr, dev, a, "student")
        _, s1, d1 = evaluate_student(st)
        log(f"  student after distillation {s1:.4f}, Delta {d1:.4f}  (teacher {t_ho:.4f})")
        res.update(student_init=how, before=s0, after=s1, delta_before=d0, delta_after=d1)
        tag = f"{a.init}_T{a.temperature:g}_e{a.epochs}"
    else:
        inits = rng.choice(pool.ids, a.students + 1, replace=False).tolist()
        shards = [rng.choice(tr, len(tr), replace=True) for _ in range(a.students)]
        probs, singles = [], []
        for j in range(a.students):
            st, how = new_student(int(inits[j]))
            distil(st, tgt, tf, teach, shards[j], dev, a, f"student {j + 1}")
            pr, s, d = evaluate_student(st)
            probs.append(pr)
            singles.append({"init": how, "score": s, "delta": d})
            log(f"  student {j + 1} ({how}): {s:.4f}, Delta {d:.4f}")
            del st
            torch.cuda.empty_cache()
        avg = np.mean(probs, axis=0)
        s_avg = score(pool, avg, y[ho])
        d_avg = float(np.sqrt(((avg - teach[ho]) ** 2).sum(1).mean()))
        st, how = new_student(int(inits[-1]))
        distil(st, tgt, tf, teach, np.unique(np.concatenate(shards)), dev, a, "union")
        _, s_u, d_u = evaluate_student(st)
        log(f"  {a.students} students averaged {s_avg:.4f} (Delta {d_avg:.4f}); one student on "
            f"the union {s_u:.4f} (Delta {d_u:.4f}); teacher {t_ho:.4f}")
        res.update(singles=singles, averaged=s_avg, delta_averaged=d_avg, union=s_u,
                   delta_union=d_u)
        tag = f"k{a.students}_e{a.epochs}"
    save(f"distill_{pool.name}_{tag}{'_smoke' if a.smoke else ''}.json", res)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ranks")
    s = sub.add_parser("soups")
    s.add_argument("--pool", required=True)
    s.add_argument("--k", type=int, default=3)
    s.add_argument("--bn-batches", type=int, default=100)
    s.add_argument("--workers", type=int, default=2)
    g = sub.add_parser("greedy-soup")
    g.add_argument("--pool", required=True)
    g.add_argument("--bn-batches", type=int, default=20)
    g.add_argument("--workers", type=int, default=2)
    n = sub.add_parser("noise")
    n.add_argument("--pool", required=True)
    n.add_argument("--sigmas", default="0.02,0.05,0.1")
    n.add_argument("--draws", type=int, default=2)
    n.add_argument("--points", type=int, default=2000)
    n.add_argument("--seed", type=int, default=0)
    n.add_argument("--workers", type=int, default=2)
    d = sub.add_parser("distill")
    d.add_argument("--pool", required=True)
    d.add_argument("--init", choices=("member", "fresh"), default="member")
    d.add_argument("--students", type=int, default=1,
                   help="k > 1: k students on bootstrap shards, each from a different "
                        "random member, plus one student on the union of the shards")
    d.add_argument("--epochs", type=int, default=12)
    d.add_argument("--temperature", type=float, default=1.0)
    d.add_argument("--lr", type=float, default=3e-4)
    d.add_argument("--batch-size", type=int, default=128)
    d.add_argument("--warmup-epochs", type=int, default=1)
    d.add_argument("--heldout-frac", type=float, default=0.2)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--smoke", action="store_true", help="a few batches only, to check the run")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    {"ranks": cmd_ranks, "soups": cmd_soups, "greedy-soup": cmd_greedy_soup,
     "noise": cmd_noise, "distill": cmd_distill}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
