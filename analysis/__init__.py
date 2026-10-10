"""The analyses that test the paper's predictions on the pools.

    pools       reading a pool: labels, compact per-member arrays, probabilities
    committees  aggregation rules, incremental committees, greedy search, scoring
    geometry    collision statistics, regime masses, Brier geometry
    predict     random-committee error curves, measured and predicted
    proposers   the label-free proposer menu P1-P7
    method      the decider on a tuning split, and its baselines
    run         everything for one pool, written to results/main/<pool>.json
    report      tables and figures for the paper from those JSON files

Nothing here writes into a pool directory.
"""
