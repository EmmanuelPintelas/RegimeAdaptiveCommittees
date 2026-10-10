# poolkit: building the pools

Everything the experiments rest on comes from these pools, so the builder is
designed around one question: can a reader trust that a pool is a fair sample of
trained models, and that nothing about the test domains shaped it?

Run everything from the repository root.

## Commands

```bash
# 1. look at the recipes a pool will draw (no GPU, no data)
python build_pool.py --config configs/pools/rxrx1.json --pool-seed 1 --dry-run 20

# 2. check every architecture fits in memory at the largest batch size
python build_pool.py --config configs/pools/rxrx1.json --preflight

# 3. a two-minute end-to-end check into a throwaway folder
python build_pool.py --config configs/pools/rxrx1.json --pool-seed 1 --smoke --stop-after 2 --out /tmp/smoke_rxrx1

# 4. the real run (default output: <BUILD_POOLS>/rxrx1_seed_1)
python build_pool.py --config configs/pools/rxrx1.json --pool-seed 1

# resume after any stop: only the pool directory is needed
python build_pool.py --out pools/rxrx1_seed_1

# progress, from any terminal, while it runs
python build_pool.py --out pools/rxrx1_seed_1 --status

# the refined (shared-ancestry) pool, after rxrx1_seed_1 has at least 20 members
python build_pool.py --config configs/pools/rxrx1.json --pool-seed 101 --parent-pool pools/rxrx1_seed_1
```

`BUILD_POOLS` is `REGADCOM_POOLS` (default `pools/` in the repository) unless the environment variable
`REGADCOM_BUILD_POOLS` points elsewhere; `--out` overrides both. `--data` gives the
dataset root if it is not `<paths.DATA>/<benchmark>`.

Planned for every benchmark `<b>` (camelyon17, kather, iwildcam, rxrx1): three
independent pools that differ in backbone as well as seed, and one refined pool.

| pool | config | seed | out |
|---|---|---|---|
| efficientnet_b0 | `<b>.json` | 1 | `pools/<b>_seed_1` |
| convnext_atto | `<b>_convnext_atto.json` | 2 | `pools/<b>_convnext_atto_seed_2` |
| mixed backbones | `<b>_mixed.json` | 3 | `pools/<b>_mixed_seed_3` |
| refined, parents from seed 1 | `<b>.json` + `--parent-pool pools/<b>_seed_1` | 101 | default |

Backbone-specific ranges follow one rule on every benchmark. efficientnet_b0 uses
the benchmark's own ranges. For convnext_atto, the learning rate runs from half
efficientnet's lower bound to a third of its upper bound, mostly with AdamW and
weight decay 1e-4 to 1e-2. For regnety_004, it runs from efficientnet's lower
bound to two thirds of its upper bound, with efficientnet's optimizer and
weight-decay choices. The mixed pool uses exactly the same per-backbone ranges as
the single-backbone pools.

Run `--preflight` on a config before its first run, while the GPU is otherwise idle.

## What makes a pool healthy

1. **Independent members.** A member's recipe and seeds depend only on the pool
   seed and its slot number. There is no adaptive search and no filter that
   compares a member with the others, so accepted members are independent draws
   from the training process. Different pool seeds give independent pools.
2. **Pre-registered.** The configuration is frozen into `pool_config.json` with a
   hash at the first run. A later run with a different configuration is refused
   (`--allow-config-change` overrides it, and the change is recorded).
3. **A minimal gate, on source validation only.** A trained draw is rejected only
   if it failed as a training run:
   - its outputs are non-finite;
   - its loss diverged, or it stayed at chance for five epochs;
   - its balanced accuracy on source validation is below 1.5× chance;
   - more of its predictions fall on one class than that class's share of the
     labels plus 0.4.

   The gate never reads a target split. It never ranks by quality, and it never
   looks at other members. Every rejected draw is committed with its reasons, and
   when it trained, its predictions are kept under `rejected/`, so every result
   can be rechecked with the gate removed. The reasoning is written out in
   `gate.py`.
4. **Targets are monitoring only.** Target-split numbers in `pool_report.json` let
   you watch a build. Nothing in the build reads them.
5. **Full provenance.** Each manifest row records:
   - the recipe, both seeds, and the parent (refined pools);
   - the per-epoch training curve and the gate checks;
   - metrics on every split and weight statistics;
   - timings, the code hash and the config hash.
6. **Shared ancestry on purpose.** A refined pool takes the first 20 accepted
   members of an independent pool as parents. It fine-tunes each into 5 children,
   each child with its own seed and a drawn short recipe. Members sharing a parent
   are the dependent case that the ancestry proposition predicts.

## Stopping and resuming

Stop at any moment: Ctrl+C, closing the window, a crash, or a power cut.

- A member is committed only when its manifest row is written, and its files are
  renamed into place just before that.
- On restart, any file of an uncommitted slot is deleted.
- That slot is trained again from scratch with the same recipe and seeds. The run
  then continues until the planned number of members is accepted.
- Only the member in progress is lost. Rejected draws are committed too, so they
  are never retried in the hope of passing.

Also on restart:
- A `.lock` file stops two processes from writing to one pool.
- Every split's labels are checked against the stored ones, and the run refuses to
  continue if the data changed.

## Layout of a pool

```
pool_config.json              frozen config + hash, code hash, environment
manifest.jsonl                one row per draw, in slot order
pool_report.json              rewritten after every draw
events.jsonl                  starts, resumes, interruptions, clean-ups
models/member_0000.pt         accepted checkpoints (loadable with netkit.models.load_member)
predictions/index.json        splits, roles, sizes, accepted ids in commit order
predictions/<split>/labels.npy
predictions/<split>/domains.npy          where the benchmark has domains
predictions/<split>/member_0000.npy      logits, float16, shape (N, K)
rejected/predictions/<split>/member_XXXX.npy
logs/run_<time>.log
```

Splits are `source_val` plus the benchmark's targets. Their role in `index.json`
is `tuning` for `ood_val*`, `test` for `ood_test*`, and `target` otherwise.

## What the analyses can rely on

- `predictions/<split>/member_XXXX.npy` holds the **pre-softmax logits** of the kept
  checkpoint on the **whole split**, row-aligned with `labels.npy`. These cover
  confidence, soft and hard votes, per-point correctness, collision statistics,
  class subsetting (restrict the logits, then renormalise), tuning-split
  subsampling, and calibration.
- Use `index.json["accepted"]` for the member list rather than globbing a folder.
- `manifest.jsonl` rows give each member's recipe, parent, weight statistics
  (`alpha`, `stable_rank`, ...), source-validation metrics (proposer P6), and
  training history.
- The full-pool ensemble, disagreement and collision numbers in the report are
  for monitoring. Recompute them from the predictions for the paper.

## Configuration

`configs/global.json` holds what every pool inherits, and is the one place to tune
the two-stage probability:

```json
"training": {"two_stage_lr_prob": 0.5, "abandon_patience": 5, "abandon_min_loss_drop": 0.01}
```

It is merged one level deep under the benchmark's own file, which wins, whenever a
pool is started or changed with `--config`; the merged result is frozen and hashed
into the pool. A resume with `--out` alone uses the configuration already frozen
there, so editing the global file never changes a pool that is under way.

### Changing a pool that is already part-built

Normally a changed configuration is refused. To apply one deliberately to the rest
of a pool, resume it once with both `--config` and `--allow-config-change`:

```bash
python build_pool.py --config configs/pools/rxrx1.json --out pools/rxrx1_seed_1 --allow-config-change
```

The change is recorded in `pool_config.json` (`config_changes`) and in
`events.jsonl`, every member records the `config_hash` it was built under, and the
report warns that the pool spans more than one configuration. Later resumes need
only `--out` again. Use it for a change to the training procedure; changing the
recipe space mid-pool would make the members stop being draws from one
distribution.

`configs/pools/<benchmark>.json`. A recipe parameter is `{"fixed": v}`,
`{"choice": [...], "weights": [...]}`, `{"uniform": [lo, hi]}`,
`{"loguniform": [lo, hi]}` or `{"int_uniform": [lo, hi]}`.

A parameter can also depend on the value another one took, which is how a
mixed-backbone pool gives each backbone its own optimisation ranges:

```json
"lr": {"by": "arch", "cases": {
  "efficientnet_b0": {"loguniform": [2e-4, 3e-3]},
  "convnext_atto":   {"loguniform": [1e-4, 1e-3]}
}}
```

The parameter it depends on must take finitely many values, and every one of them
needs a case. A case cannot itself be conditional.

Required parameters:
- **`recipe`:** `arch`, `pretrained`, `augment`, `optimizer`, `lr`, `weight_decay`,
  `batch_size`, `epochs`, `scheduler` (`plateau`, `halve_each_epoch`, `none`),
  `sched_patience`, `sched_factor`, `stop_patience`, `label_smoothing`,
  `drop_rate`, `train_frac`.
- **`refine.recipe`:** the same list without `arch`, `pretrained` and `drop_rate`.

Augmentation names are listed at the top of `augment.py`.

An optional `training` block controls when a run is given up on:

```json
"training": {"abandon_patience": 8, "abandon_min_loss_drop": 0.01}
```

A run is abandoned only when it has been at chance on source validation for
`abandon_patience` epochs (default 5) *and* its training loss has not fallen by
`abandon_min_loss_drop` (default 0.01, i.e. 1%) over that streak. Validation
alone cannot tell a slow starter from a dead run on a split with many classes;
the loss can, since a dead run sits at log(K). Raise the patience for a backbone
that needs a long warm-up, as `rxrx1_convnext_atto.json` does.

The same block carries `two_stage_lr_prob` (0.5 in every shipped config, 0 when
the key is absent). With that probability a member trains at the **top** of its
learning-rate range for the first `ceil(epochs / 2)` epochs and at the **bottom**
for the rest. In a mixed pool the two rates are the member's own backbone's
range. While it is on, the drawn rate and the drawn scheduler are not used, and
early stopping is held back until after the switch so the first stage always runs
to its end. The flag is drawn from its own stream, so changing the probability
moves no recipe draw, and refined pools never use it. Every member records what
it did in `manifest.jsonl` under `schedule`.

Review a configuration before the first run of a pool. After that it is frozen.
To change it, start a new pool.

Storage per accepted member is about `2 × K × (N_source_val + N_targets)` bytes
of logits plus the checkpoint.
