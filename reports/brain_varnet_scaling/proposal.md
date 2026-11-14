# Proposal: brain VarNet model-size × data-size scaling (R=8)

**Status: approved by the user on 2026-11-12 ("approved").** Choices the user made while drafting:
- cascades 2, 4, 6, 8 and 12, with channel width left to a follow-up;
- an equal step budget of 100 epochs at 80 volumes;
- seed 43 for the top two cascade sizes at each data level.

- Topic / branch: `brain_varnet_scaling`, branch `exp/brain_varnet_scaling`,
  worktree `../SeqMRI-worktrees/exp-brain_varnet_scaling`, report folder `reports/brain_varnet_scaling/`.
- Roadmap: step 1 (anatomy routing). It supports the brain static-vs-learned question of `exp/brain_static_vs_loupe`,
  whose comparisons are limited by overfitting.

## Question
Brain-only VarNets on 20 volumes overfit. In `brain_static_vs_loupe`'s 130-epoch runs (seed 42), val peaks around
epoch 65 and then loses 0.7–0.9 dB while train loss keeps falling.

1. At 20 volumes, does a smaller VarNet (fewer cascades) reach a higher val peak than the 12-cascade default?
2. Does the best cascade count grow with training data (20 → 40 → 80 volumes)?

The best size per data level then becomes the reconstructor for a rerun of static vs learned masks. Channel width is a
later follow-up.

## Design
The mask is frozen to S, as in brain_static_vs_loupe's static arm:
`loss_domain/runs_v2/loss_domain_rss_r8_flair_knee_100ep_seed42/best_model.pt` in all three
`sampler.fixed_mask_checkpoints` slots.

Everything else matches static_s42:
- brain-only training (`datasets.train_anatomies=[brain]`), R=8 of acquired (`sampler.budget_over=acquired`);
- Muon + LayerNorm, L1 on RSS, batch 16, lr 1e-3, `sampler.anatomy_conditioning=true`, `recon.chans=18`.

| Factor | Levels |
|---|---|
| `recon.num_cascades` | 2, 4, 6, 8, 12 |
| `datasets.limit_volumes` | 0.25 / 0.5 / 1.0 (20 / 40 / 80 brain train volumes) |
| Seed | 42 for all 15 cells; 43 for the top two cascade sizes at each data level (up to 6 runs) |

**Equal step budget**: every run gets the optimiser steps of 100 epochs at 80 volumes.

| Volumes | Epochs | lr drop at epoch | Val every |
|---|---|---|---|
| 80 | 100 | 90 | 5 epochs |
| 40 | 200 | 180 | 10 epochs |
| 20 | 400 | 360 | 20 epochs |

- The lr drops 10× at 90% of the steps. Val runs at the same step points in every run (20 evals).
- Before launch, the exact slice counts set the epoch counts so that the steps match within 1%. The run-state commit
  records the final numbers.

**Peak checkpoint**: a new option `datasets.val_anatomies=[brain]` restricts the trainer's val set (and so its val loss
and `best_model.pt`) to brain. Without it, the best checkpoint is picked by a val loss that includes knee, which these
models never train on. It ships with a unit test in the run-state commit.

## Scoring
- Each run's `best_model.pt` (peak) and `last_model.pt` are scored on the full brain val set (40 volumes, 605 slices)
  with S.
- Same common per-slice Bernoulli draws (seeds 1234, 4321), trainer apodized metric mask and per-slice pairing as
  `brain_static_vs_loupe`. CIs bootstrap over volumes.
- Each run is also scored on its own training slices, for the train − val gap.

## Figures and readouts
- PSNR against cascade count, one line per data size, at the peak checkpoint. Also SSIM and NMSE.
- PSNR against training volumes (log scale), one line per cascade count.
- Train − val gap against cascades. Val curves against steps for every run. Peak step per run.
- Parameter count and wall time per step for each cascade count.

## Expected outcome
- 20 volumes: 4–8 cascades peak 0.1–0.3 dB above 12, with a smaller train − val gap.
- 80 volumes: 12 cascades are best or tied.
- 12-cascade peak gain from 20 → 80 volumes: +0.8 to +1.3 dB.

## Noise floor
- Repeating a run at a fixed seed moves it about 0.03 dB. Static's seed spread is 0.44 dB.
- So a size effect counts only when it holds, paired per slice, in both s42 and s43.

## Criteria (full-val PSNR at the peak checkpoint)
- **Validity:** every run finishes, brain SSIM ≥ 0.80.
- **Q1 at 20 volumes:**
  - smaller wins: the best smaller size beats 12 cascades by ≥ +0.15 dB in both seeds, with the CI above 0 in each;
  - capacity not the limit: |difference| < 0.05 dB in both seeds;
  - bigger wins: 12 cascades beat every smaller size by ≥ 0.15 dB in both seeds;
  - otherwise inconclusive.
- **Q2:** the best cascade count at 80 volumes is larger than at 20 volumes in both seeds → "optimum shifts". Otherwise
  "no shift".
- **Top two sizes for seed 43** (per data level): the two highest seed-42 peak PSNRs. When 12 cascades is not among
  them, it also gets a seed-43 run at 20 volumes, so Q1 has its paired control.
- **Next step:**
  - smaller wins: rerun static vs learned masks (S, M, U) with that size at 20 volumes;
  - otherwise: go to full data with 12 cascades.

## Compute
- About 7,600 steps per run. A 12-cascade run takes about 3.7 h; 2–8 cascades take about 0.3–0.7× that. A smoke test
  times one epoch of each size before launch.
- Seed-42 grid: 15 runs, about 33 GPU-h, about 17 h on GPUs 0–1.
- Seed-43 round: up to 7 runs, about 15 GPU-h.
- GPUs 0–1 only. Launch after `brain_static_vs_loupe`'s long-schedule runs and their scoring. Crashed runs are rerun in
  full.
