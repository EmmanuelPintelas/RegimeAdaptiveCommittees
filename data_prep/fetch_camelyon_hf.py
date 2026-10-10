"""Fetch the Camelyon17 mirror and check it against the official WILDS counts.

The CodaLab host serving the official WILDS archives has been down.  The
HuggingFace mirror ``wltjr1007/Camelyon17-WILDS`` is a complete copy of the
dataset -- all 455,954 patches -- repackaged as parquet under three split names
rather than four.  Only the packaging differs, and row arithmetic pins it down:

    mirror train       302,436  = WILDS train                   (identical)
    mirror validation   68,464  = WILDS id_val (33,560) + val (34,904)
    mirror test         85,054  = WILDS test                    (identical)

Every row carries its ``center``, so the official four-way protocol is recovered
exactly by filtering on hospital, which ``pack_camelyon.py`` does and refuses to
proceed unless all four counts match.  No hospital is missing and no split is
approximated; the experiments run on the official protocol.

Parquet is left as parquet.  Materialising 455,000 patches as loose files would
cost another ten gigabytes and a great deal of filesystem churn for nothing.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = "wltjr1007/Camelyon17-WILDS"
EXPECT = {"train": 302436, "validation": 68464, "test": 85054}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="../data/camelyon17_hf")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    from huggingface_hub import snapshot_download
    import shutil
    free = shutil.disk_usage("C:/").free / 1e9
    print(f"free disk {free:.0f} GB, need about 11 GB")
    if free < 15:
        print("not enough headroom; free some space first")
        return 1

    print(f"downloading {REPO} -> {os.path.abspath(a.root)}")
    path = snapshot_download(REPO, repo_type="dataset", local_dir=a.root,
                             max_workers=a.workers)
    print(f"\nsnapshot at {path}")

    import pyarrow.parquet as pq
    import glob
    total = {}
    for split in EXPECT:
        files = sorted(glob.glob(os.path.join(path, "data", f"{split}-*.parquet")))
        n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        total[split] = n
        flag = "ok" if n == EXPECT[split] else f"EXPECTED {EXPECT[split]}"
        print(f"  {split:11} {len(files):>2} shards  {n:>7} rows  {flag}")

    if total != EXPECT:
        print("\nthe mirror does not match the counts this script was written "
              "against; check before relying on it")
        return 1
    print("\ncounts match the characterisation above")
    print("next:\n  python pack_camelyon.py --src " + a.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
