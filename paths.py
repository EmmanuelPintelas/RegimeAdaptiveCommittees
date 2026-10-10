"""Where the code reads and writes data.

Every location can be set with an environment variable, so nothing needs editing:

    REGADCOM_POOLS        the downloaded or built pools, one folder per pool
                          (default: <repository>/pools)
    REGADCOM_DATA         packed datasets, needed only to train members or run a forward
                          pass (default: <repository>/data)
    REGADCOM_BUILD_POOLS  where build_pool.py, build_replica.py and build_public_pool.py
                          write new pools (default: REGADCOM_POOLS)

Bash:        export REGADCOM_POOLS=/path/to/pools
PowerShell:  $env:REGADCOM_POOLS = "D:\\pools"

Results, tables and figures are written under <repository>/results, and per-pool
caches under <repository>/cache.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

POOLS = os.environ.get("REGADCOM_POOLS", os.path.join(ROOT, "pools"))
DATA = os.environ.get("REGADCOM_DATA", os.path.join(ROOT, "data"))
BUILD_POOLS = os.environ.get("REGADCOM_BUILD_POOLS", POOLS)

RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "cache")
