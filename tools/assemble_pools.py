"""Check and unpack downloaded pool parts into the pools folder.

Download any subset of the release (a whole benchmark folder, or single pools) into one
folder, keeping or flattening its sub-folders, together with SHA256SUMS.txt, then:

    python tools/assemble_pools.py --downloads /path/to/downloads
    python tools/assemble_pools.py --downloads D:/RegAdCom --pools D:/pools --no-verify

Every part is a normal zip whose entries start with "<pool>/", named
<pool>.predictions.<i>of<n>.zip or <pool>.models.<i>of<n>.zip.  The script checks each
part against SHA256SUMS.txt (when present), unpacks it into the pools folder (default
REGADCOM_POOLS, see paths.py), and reports, per pool, whether all predictions parts
(enough for every analysis) and all model parts (checkpoints) are in place.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

PART = re.compile(r"^(?P<pool>.+)\.(?P<kind>predictions|models)\.(?P<i>\d+)of(?P<n>\d+)\.zip$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 * 2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def pool_status(root):
    """Which of a pool's predictions and checkpoints are on disk."""
    import json
    index = os.path.join(root, "predictions", "index.json")
    if not os.path.exists(index):
        return "predictions: index missing (first predictions part not unpacked)"
    with open(index, encoding="utf-8") as f:
        idx = json.load(f)
    ids = [int(m) for m in idx["accepted"]]
    missing = sum(not os.path.exists(os.path.join(root, "predictions", s, f"member_{m:04d}.npy"))
                  for s in idx["splits"] for m in ids)
    pred = "predictions: complete" if not missing else f"predictions: {missing} files missing"
    models_dir = os.path.join(root, "models")
    if not os.path.isdir(models_dir):
        return pred + "; models: not downloaded"
    absent = sum(not os.path.exists(os.path.join(models_dir, f"member_{m:04d}.pt")) for m in ids)
    return pred + ("; models: complete" if not absent else f"; models: {absent} missing")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--downloads", required=True, help="folder holding the downloaded parts")
    ap.add_argument("--pools", default=paths.POOLS, help="where to unpack (default REGADCOM_POOLS)")
    ap.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check")
    ap.add_argument("--keep", action="store_true", help="keep the zip parts after unpacking")
    a = ap.parse_args(argv)

    sums = {}
    for root, _, files in os.walk(a.downloads):
        if "SHA256SUMS.txt" in files:
            for line in open(os.path.join(root, "SHA256SUMS.txt"), encoding="utf-8"):
                if line.strip():
                    digest, name = line.split(maxsplit=1)
                    sums[os.path.basename(name.strip())] = digest
    found = []
    for root, _, files in os.walk(a.downloads):
        for f in sorted(files):
            m = PART.match(f)
            if m:
                found.append((os.path.join(root, f), m))
    if not found:
        print(f"no parts found under {a.downloads}")
        return 1

    os.makedirs(a.pools, exist_ok=True)
    bad = []
    for path, m in found:
        name = os.path.basename(path)
        if not a.no_verify:
            if name not in sums:
                print(f"  {name}: not in SHA256SUMS.txt, unpacking unverified")
            elif sha256(path) != sums[name]:
                print(f"  {name}: CHECKSUM MISMATCH, download it again")
                bad.append(name)
                continue
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                if not member.startswith(m["pool"] + "/") or ".." in member.split("/"):
                    raise SystemExit(f"{name}: unexpected entry {member!r}")
            z.extractall(a.pools)
        print(f"  unpacked {name}")
        if not a.keep:
            os.remove(path)

    print(f"\npools in {a.pools}:")
    for pool in sorted(os.listdir(a.pools)):
        print(f"  {pool:34s} {pool_status(os.path.join(a.pools, pool))}")
    if bad:
        print(f"\n{len(bad)} part(s) failed the checksum: {', '.join(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
