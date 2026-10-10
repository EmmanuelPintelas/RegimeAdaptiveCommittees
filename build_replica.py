"""Build a replica of an existing pool: same recipe law, new pool seed.

A reference pool may have been built under several configuration versions, recorded in
its pool_config.json (the last version plus the list of changes) and in its manifest
(the version each member was drawn under).  This script rebuilds every version from
that record, checks each against its recorded hash, and builds the replica in the same
blocks: as many accepted members under each version, in the same order, as the
reference has.  Members are drawn with the replica's own pool seed, so the replica is an
independent pool from the same recipe law.

    python build_replica.py --reference pools/camelyon17_seed_1 --pool-seed 4 --check
    python build_replica.py --reference pools/camelyon17_seed_1 --pool-seed 4

Stop it at any time; running the same command again resumes where it stopped (the
member being trained is discarded and trained again, as with build_pool.py).
"""
from __future__ import annotations

import argparse
import ast
import copy
import itertools
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import paths  # noqa: E402
from build_pool import normalize, science  # noqa: E402
from poolkit import recipes as R  # noqa: E402
from poolkit.recipes import digest  # noqa: E402


def set_flat(cfg, key, value):
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    if value == "<absent>":
        d.pop(parts[-1], None)
    else:
        d[parts[-1]] = value


def parse_change(line):
    key, rest = line.split(": ", 1)
    old, new = rest.split(" -> ")
    return key, ast.literal_eval(old), ast.literal_eval(new)


def versions_of(reference):
    """{hash: config} for every configuration version the reference pool was built under.

    Returns also the versions whose recorded hash cannot be recomputed (a version created
    by an older code release, which hashed differently); those are verified by redrawing
    their members' recipes instead (see recipes_match)."""
    with open(os.path.join(reference, "pool_config.json"), encoding="utf-8") as f:
        frozen = json.load(f)
    if digest(science(frozen["config"])) != frozen["config_hash"]:
        raise SystemExit("the reference pool's frozen configuration does not match its hash")
    cfg = copy.deepcopy(frozen["config"])
    out, unhashed = {frozen["config_hash"]: copy.deepcopy(cfg)}, []
    for change in reversed(frozen.get("config_changes") or []):
        for line in change["changes"]:
            key, old, _ = parse_change(line)
            set_flat(cfg, key, old)
        if digest(science(cfg)) != change["from"]:
            unhashed.append(change["from"])
        out[change["from"]] = copy.deepcopy(cfg)
    return out, unhashed


def recipes_match(cfg, rows):
    """Do the reference members' recipes redraw exactly from this configuration?"""
    for r in rows:
        drawn = R.draw_recipe(cfg["recipe"], cfg["pool_seed"], int(r["member_id"]))
        rec = r["recipe"]
        if set(drawn) != set(rec):
            return False
        for k, v in drawn.items():
            w = rec[k]
            if isinstance(v, float) or isinstance(w, float):
                if abs(float(v) - float(w)) > 1e-9 * max(1.0, abs(float(w))):
                    return False
            elif v != w:
                return False
    return True


def blocks_of(reference):
    """[(config hash, accepted reference rows)] in member order."""
    rows = [json.loads(line) for line in open(os.path.join(reference, "manifest.jsonl"),
                                              encoding="utf-8") if line.strip()]
    rows.sort(key=lambda r: int(r["member_id"]))
    acc = [r for r in rows if r["status"] == "accepted"]
    return [(h, list(g)) for h, g in itertools.groupby(acc, key=lambda r: r["config_hash"])]


def accepted(out, any_status=False):
    """Accepted members of the replica so far (every committed draw with any_status)."""
    path = os.path.join(out, "manifest.jsonl")
    if not os.path.exists(path):
        return 0
    return sum(1 for line in open(path, encoding="utf-8")
               if line.strip() and (any_status or json.loads(line)["status"] == "accepted"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True, help="the pool to replicate")
    p.add_argument("--pool-seed", type=int, required=True)
    p.add_argument("--out", help="default <BUILD_POOLS>/<benchmark>_seed_<seed>")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data", help="dataset root, as for build_pool.py")
    p.add_argument("--check", action="store_true",
                   help="rebuild and verify the configurations, print the plan, and exit")
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end test: one member under each of the first two "
                        "configurations, 3 batches per epoch, into smoke_<benchmark>_seed_<seed>")
    args = p.parse_args(argv)

    reference = os.path.abspath(args.reference)
    versions, unhashed = versions_of(reference)
    blocks = blocks_of(reference)
    missing = [h for h, _ in blocks if h not in versions]
    if missing:
        raise SystemExit(f"members drawn under unrecorded configurations: {missing}")
    for h, rows in blocks:
        if not recipes_match(versions[h], rows):
            raise SystemExit(f"the rebuilt configuration {h} does not redraw its members' "
                             "recipes; refusing to build a replica from it")
    for h in unhashed:
        print(f"note: configuration {h} predates the current hashing; verified by redrawing "
              "its members' recipes")
    bench = versions[blocks[0][0]]["benchmark"]
    name = ("smoke_" if args.smoke else "") + f"{bench}_seed_{args.pool_seed}"
    out = os.path.abspath(args.out or os.path.join(paths.BUILD_POOLS, name))
    if args.smoke:
        blocks = [(h, rows[:1]) for h, rows in blocks[:2]]
    if os.path.abspath(out) == reference:
        raise SystemExit("the replica would overwrite the reference pool")

    cfg_dir = os.path.join(out, "replica_configs")
    os.makedirs(cfg_dir, exist_ok=True)
    plan, total = [], 0
    for i, (h, rows) in enumerate(blocks):
        n = len(rows)
        cfg = copy.deepcopy(versions[h])
        cfg["pool_seed"] = args.pool_seed
        again = normalize(copy.deepcopy(cfg))
        if digest(science(again)) != digest(science(cfg)):
            raise SystemExit(f"configuration {h} does not survive normalization unchanged")
        path = os.path.join(cfg_dir, f"v{i + 1}_{h}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
        total += n
        plan.append((path, h, n, total))

    print(f"reference {reference}\nreplica   {out}  (pool seed {args.pool_seed})")
    for path, h, n, upto in plan:
        print(f"  {n:3d} members under {h}  (members up to {upto})  {os.path.basename(path)}")
    done = accepted(out, args.smoke)
    print(f"accepted so far: {done}")
    if args.check:
        return 0

    for path, h, n, upto in plan:
        while accepted(out, args.smoke) < upto:
            todo = upto - accepted(out, args.smoke)
            cmd = [sys.executable, os.path.join(HERE, "build_pool.py"), "--config", path,
                   "--pool-seed", str(args.pool_seed), "--out", out, "--global-config", "",
                   "--allow-config-change", "--stop-after", str(todo),
                   "--workers", str(args.workers)]
            if args.data:
                cmd += ["--data", args.data]
            if args.smoke:
                cmd += ["--smoke"]
            print(f"\n=== configuration {h}: {accepted(out, args.smoke)}/{upto} accepted, "
                  f"training up to {todo} more")
            code = subprocess.call(cmd, cwd=HERE)
            if code != 0:
                print(f"build_pool.py exited with {code}; run this command again to resume")
                return code
    print(f"\nreplica complete: {accepted(out, args.smoke)} members")
    return 0


if __name__ == "__main__":
    sys.exit(main())
