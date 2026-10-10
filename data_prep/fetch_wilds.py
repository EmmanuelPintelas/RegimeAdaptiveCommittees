"""Download a WILDS task, waiting out the host if it is unavailable.

The WILDS distribution is served from CodaLab, which has been returning HTTP 500
for every dataset.  Rather than fail once, this polls the endpoint and starts the
download the moment it answers, so the transfer begins unattended.

    python fetch_wilds.py --task camelyon17 --root ../data/wilds
    python fetch_wilds.py --task camelyon17 --root ../data/wilds --check-only

Resumes are handled by the wilds package itself: if a previous attempt left a
partial archive, delete the task directory and start again.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import urllib.request
import warnings

warnings.filterwarnings("ignore")


def download_url(task):
    mod = importlib.import_module(f"wilds.datasets.{task}_dataset")
    cls = [v for k, v in vars(mod).items()
           if k.endswith("Dataset") and hasattr(v, "_versions_dict")][0]
    versions = cls._versions_dict
    latest = versions[list(versions)[-1]]
    return latest.get("download_url", ""), latest.get("compressed_size")


def available(url, timeout=45):
    """Is the host actually serving bytes?

    A HEAD request is not enough, because CodaLab answers 500 to HEAD on
    endpoints that would serve a GET.  Nor can the probe ask for a byte range:
    CodaLab rejects a Range header outright with 400, so a ranged probe reports
    the host as down at the very moment it is serving perfectly well, and a
    poller built on one waits for a recovery that has already happened.  The
    probe is therefore a plain GET whose first kilobyte is read and then
    abandoned, which is also the reason a broken transfer here cannot be
    resumed and has to start again.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read(1024)
            if r.status == 200 and not body.lstrip().startswith(b"<"):
                return True, f"{r.status}, {len(body)} bytes of payload"
            return False, f"{r.status} but the body looks like an error page"
    except Exception as e:
        return False, f"{type(e).__name__}: {getattr(e, 'code', e)}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="camelyon17")
    p.add_argument("--root", default="../data/wilds")
    p.add_argument("--check-only", action="store_true",
                   help="report whether the host is serving and exit")
    p.add_argument("--poll-minutes", type=float, default=20.0)
    p.add_argument("--max-hours", type=float, default=12.0)
    a = p.parse_args(argv)

    url, size = download_url(a.task)
    gb = (size or 0) / 1e9
    print(f"{a.task}: {gb:.1f} GB compressed", flush=True)
    print(f"  {url}", flush=True)

    ok, why = available(url)
    print(f"  host check: {'SERVING' if ok else 'unavailable'} ({why})", flush=True)
    if a.check_only:
        return 0 if ok else 1

    deadline = time.time() + a.max_hours * 3600
    attempt = 0
    while not ok:
        if time.time() > deadline:
            print(f"\ngiving up after {a.max_hours:g} h; the host never answered",
                  flush=True)
            return 1
        attempt += 1
        mins = a.poll_minutes
        print(f"  attempt {attempt} failed, retrying in {mins:g} min "
              f"({(deadline - time.time()) / 3600:.1f} h left)", flush=True)
        time.sleep(mins * 60)
        ok, why = available(url)
        if ok:
            print(f"  host is serving again ({why})", flush=True)

    os.makedirs(a.root, exist_ok=True)
    free = __import__("shutil").disk_usage(os.path.abspath(a.root)).free / 1e9
    need = gb * 2.0
    print(f"  free space {free:.0f} GB, need roughly {need:.0f} GB "
          "(archive plus extraction)", flush=True)
    if free < need:
        print("  not enough room; free some space and rerun", flush=True)
        return 1

    from wilds import get_dataset
    print(f"\ndownloading to {os.path.abspath(a.root)}", flush=True)
    ds = get_dataset(dataset=a.task, root_dir=a.root, download=True)
    print(f"\ndone: {len(ds)} examples, {ds.n_classes} classes", flush=True)
    for split in ("train", "id_val", "val", "test"):
        try:
            print(f"  {split:8} {len(ds.get_subset(split))}", flush=True)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
