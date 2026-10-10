"""Building pools of trained members that later analyses can rest on.

    recipes   the frozen recipe distribution and the per-member draw
    augment   named augmentation pipelines, with no module-level state
    train     training one member, selected on source validation only
    gate      the pre-registered acceptance gate
    weights   data-free weight statistics recorded at commit
    metrics   fast decision metrics (recall, F1, geometric mean, ECE, NLL)
    store     on-disk layout, atomic writes, manifest, lock, resume clean-up
    report    the pool state and pool_report.json, rewritten after every member

The entry point is build_pool.py, one directory up.  README.md in this folder
describes the layout of a pool and the contract its predictions follow.
"""
