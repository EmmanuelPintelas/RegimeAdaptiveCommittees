# RegAdCom: How Many Models, and Which?

Code, results and model pools for **"How Many Models, and Which? Regime-Adaptive
Committee Selection under Distribution Shift"** (Emmanuel Pintelas).

Deploying a pool of trained models under distribution shift means choosing which
members to combine and how many. The paper gives an exact account of plurality
committees: a committee errs when its strongest wrong label collects as many votes as
the true one, so a random committee's error is governed by how often wrong members make
the same mistake (the collision spectrum), and what selection can gain over the full pool
is bounded by how much of the pool's mistakes its members get right. This separates a
collision regime, where a few members beat the full pool, from a scatter regime, where
the full pool is hard to beat. On top of the theory, label-free proposers and a decider
with a regime prior and a tuning-domain guarantee choose a committee and its size.

This repository lets you

* regenerate every table and figure of the paper from the stored results, with no
  download and no GPU;
* rerun every analysis from the released pools (22 pools, the predictions of 2,144
  trained or public models on four shifted image benchmarks and an ImageNet-R
  benchmark);
* run the method on your own pool, or build new pools with the same recipe laws.

## Contents

```
analysis/        every analysis of the paper (python -m analysis.<module>)
poolkit/         pool building: recipes, training, acceptance gate, storage (see poolkit/README.md)
netkit/          datasets, augmentations and the member network
configs/         the recipe distribution of every pool
build_pool.py    builds a pool of independently trained members (resumable)
build_replica.py rebuilds a pool's recipe law with new seeds
build_public_pool.py  pools of public checkpoints on the imagenet_r200 benchmark
appendix_b.py    the alternatives of Appendix B (soups, distillation, rank averaging)
data_prep/       download and pack the four benchmarks
tools/           unpack downloaded pools
results/         every result the paper reports (main, frozen, appendix_b), its tables and figures
cache/phase3/    per-pool weight-noise scores (computed from checkpoints and images)
paths.py         where pools and data live (environment variables)
```

## Install

Python 3.10 or later and PyTorch with CUDA are recommended (the analyses run on CPU, slowly).

```bash
pip install -r requirements.txt
```

## 1. Regenerate the paper's tables and figures (no download)

```bash
python -m analysis.report          # tables and figures from results/main
python -m analysis.frozen --table  # the frozen-evaluation table from results/frozen
```

Tables are written to `results/tables/` (LaTeX) and figures to `results/figures/` (PDF).
No number in them is typed by hand.

## 2. Download the pools

**Link: to be added.**

The release is one folder per benchmark. Each pool comes as standalone zip parts:

* `<pool>.predictions.<i>of<n>.zip`: the stored logits of every member on every split,
  with the pool's metadata. This is all every analysis needs.
* `<pool>.models.<i>of<n>.zip`: the member checkpoints, needed only to run members on
  new images or to recompute weight-based signals.

| benchmark | pools | predictions | checkpoints |
|---|---|---|---|
| camelyon17 | 6 | 0.4 GB | 9.4 GB |
| kather | 4 | 0.2 GB | 6.2 GB |
| iwildcam | 4 | 9.5 GB | 6.4 GB |
| rxrx1 | 6 | 71.6 GB | 12.3 GB |
| imagenet_r200 (public checkpoints) | 2 | 2.7 GB | (published elsewhere) |

Pools per benchmark: `<b>_seed_1` (EfficientNet-B0), `<b>_convnext_atto_seed_2`,
`<b>_mixed_seed_3` (three backbones), `<b>_refined_seed_101` (five children of each of
twenty members of `<b>_seed_1`), and for Camelyon17 and RxRx1 the replicas `<b>_seed_4`
and `<b>_seed_5`. `public_clip_soups` and `public_timm_in1k` hold the predictions of 72
public CLIP fine-tunes and 72 public timm ImageNet-1k models.

Download what you need together with `SHA256SUMS.txt`, then check and unpack:

```bash
python tools/assemble_pools.py --downloads /path/to/downloads   # into ./pools
```

Pools go to `./pools` unless `REGADCOM_POOLS` points elsewhere (see `paths.py`).

## 3. Rerun the analyses from the pools

```bash
python -m analysis.run --all              # per-pool analyses -> results/main/<pool>.json
python -m analysis.crossfit --all         # cross-fitted greedy references
python -m analysis.bootstrap --all        # paired cluster-bootstrap intervals
python -m analysis.regime_map             # Figure 3
python -m analysis.ancestry               # refined pools (Table 7)
python -m analysis.signal_ablation        # Shapley values of the consensus signals
python -m analysis.tuning_size            # Figure 4
python -m analysis.score_ablation         # accuracy instead of balanced accuracy
python -m analysis.pure_noise             # Table 1
python -m analysis.report                 # tables and figures

python -m analysis.frozen public_clip_soups public_timm_in1k camelyon17_seed_4 camelyon17_seed_5 rxrx1_seed_4 rxrx1_seed_5
python -m analysis.frozen --table
```

`analysis.run` reads the weight-noise scores from `cache/phase3/<pool>/`; to recompute them
(checkpoints and images needed) run `python -m analysis.weight_noise --all`. RxRx1 pools
need the most memory: the average of probabilities streams members from host memory.

## 4. Use the method on your own pool

A pool is a folder in the layout `build_pool.py` writes (described in
`poolkit/README.md`):

```
<pool>/pool_config.json        {"config": {...}}
<pool>/manifest.jsonl          one row per member: member_id, arch, metrics per split
                               (metrics.source_val is the source-validation signal),
                               weight_stats (the spectral exponent "alpha")
<pool>/predictions/index.json  benchmark, num_classes, accepted member ids, and the
                               splits with their roles ("tuning", "test", "source_val")
<pool>/predictions/<split>/labels.npy, domains.npy, member_XXXX.npy   (N, K) logits
<pool>/models/member_XXXX.pt   checkpoints (netkit.models.save_member format)
```

The released pools are complete examples. Two of the label-free signals read member
weights: the spectral exponent (stored in the manifest when a member is committed, see
`poolkit/weights.py`) and weight-noise stability, which
`python -m analysis.weight_noise path/to/pool` computes from the checkpoints and the
benchmark's test images (a loader in `netkit/data.py`). Then
`python -m analysis.run path/to/pool` runs the decider (Algorithm 1, in
`analysis/method.py`), the proposers (`analysis/proposers.py`) and every analysis of the
paper on your pool; the committee machinery is in `analysis/committees.py`.

## 5. Build pools from scratch

```bash
cd data_prep
python fetch_wilds.py --task iwildcam --root ../data/wilds
python pack_wilds.py --task iwildcam --src ../data/wilds --dst ../data/iwildcam
python pack_kather.py --src ../data/kather_raw --dst ../data/kather
#   (NCT-CRC-HE-100K.zip and CRC-VAL-HE-7K.zip from Zenodo record 1214456 in data/kather_raw)
python fetch_camelyon_hf.py && python pack_camelyon.py
cd ..
python build_pool.py --config configs/pools/iwildcam.json --pool-seed 1
python build_pool.py --config configs/pools/iwildcam.json --pool-seed 101 --parent-pool pools/iwildcam_seed_1
python build_replica.py --reference pools/rxrx1_seed_1 --pool-seed 4
python build_public_pool.py --pool clip_soups
```

Building is resumable and every member's recipe, seeds and code version are recorded;
`poolkit/README.md` describes the acceptance gate and the stored layout.

## Data and licenses

The code is released under the MIT license (`LICENSE`). The pools contain predictions on,
and models trained on, Camelyon17, iWildCam and RxRx1 (WILDS), the Kather colorectal
cohorts (Zenodo record 1214456), and ImageNet-V2, ImageNet-R, ImageNet-Sketch and
ImageNet-A, and predictions of the model-soups CLIP checkpoints and of timm models; the
terms of those datasets and checkpoints apply to anything derived from them.

## Citation

Citation details will be added when the paper is public; see `CITATION.cff`.
