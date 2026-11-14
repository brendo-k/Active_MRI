# Proposal: brain VarNet cascades × channels (R=8)

**Status: approved by the user on 2026-11-14** ("Okay this is good, lets train and run the scaling experiment. You can
run on all gpus here."). It follows `reports/brain_varnet_scaling/report.md` (merged as `9fad53f`), whose verdict
asked for channels as the second dimension.

- Topic / branch: `brain_varnet_width`, branch `exp/brain_varnet_width`, worktree
  `../SeqMRI-worktrees/exp-brain_varnet_width`, report folder `reports/brain_varnet_width/`.
- Roadmap: step 1 (anatomy routing). It picks the reconstructor for the rerun of static vs learned brain masks.

## Question
At 80 training volumes, adding cascades stopped helping at about 6 in `brain_varnet_scaling`.
1. Does making each cascade wider (more U-Net channels) help instead?
2. For the same parameter count, is it better to spend parameters on more cascades or on wider ones?

## Design
Everything matches the 80-volume runs of `brain_varnet_scaling`; only `recon.num_cascades` and `recon.chans` vary.
- Brain FLAIR, local v1 manifest tier S, all 80 training volumes (1,211 slices; `datasets.limit_volumes=1.0`).
- Frozen mask S (`loss_domain/runs_v2/loss_domain_rss_r8_flair_knee_100ep_seed42/best_model.pt`), R=8 of acquired.
- Muon + LayerNorm, L1 on RSS, batch 16, lr 1e-3 dropping 10× at epoch 90 of 100.
- 7,600 steps (100 epochs × 76 steps); val every 5 epochs on the pinned 10-volume brain subset; early-stop patience 5.

**Why 80 volumes:** seed noise is smallest there (0.02–0.09 dB between seeds), overfitting is mild, and at 20 volumes
every size overfits by step about 1,900, so capacity barely matters there.

**Grid.** `recon.chans` is the first U-Net level's width; it doubles at each of the 4 pooling levels, so parameters per
cascade grow with the square of `chans`.
- Channels {12, 18, 26, 36}: steps of about √2, so parameters per cascade roughly double (1.1, 2.5, 5.1, 9.8 M).
- Cascades {3, 6, 12}: doubling too. So cells on one diagonal have nearly equal parameter counts.

| Total parameters (M) | 3 cascades | 6 cascades | 12 cascades |
|---|---|---|---|
| 12 channels | 4.1 | 7.4 | 14.0 |
| 18 channels | 8.2 | 15.5 (reused) | 30.3 (reused) |
| 26 channels | 16.2 | 31.6 | 62.3 |
| 36 channels | 30.3 | 59.7 | 118.7 |

- Equal-size groups: about 15 M (3×26, 6×18, 12×12), about 30 M (3×36, 6×26, 12×18), about 60 M (6×36, 12×26).
- Reused: `brain_varnet_scaling/c6_v80_s42`, `c6_v80_s43`, `c12_v80_s42` and `c12_v80_s43` (same code apart from early
  stopping, which c12_v80_s42 ran without; its peak checkpoint does not depend on that).
- Left out: 48 channels (12×48 runs out of memory even with checkpointing; a wide-only probe has no equal-size partner);
  2 and 4 cascades (already known to lose 0.2–1.0 dB at 18 channels).

**Benchmark** (2026-11-14, GPU 0, batch 16, 3 short epochs; relative to 12×18):

| Cell | Relative step time | Memory |
|---|---|---|
| 3×36 | 0.68× | fits |
| 6×36 | 0.90× | fits |
| 12×26 | 1.12× | fits with no headroom → run with `recon.checkpoint_cascades=true` (1.51×) |
| 12×36 | — | out of memory → `recon.checkpoint_cascades=true` (1.82×) |

Cascade checkpointing recomputes activations in the backward pass; it changes speed and memory, not results.

**Learning rate:** lr 1e-3 for every cell. Muon normalizes each weight update's size, which should transfer across
widths. The widest model (118.7 M) must train without NaNs or skipped steps.

## Runs
- Seed 42: the 10 new cells, about 23 GPU-hours.
- Seed 43: always 3×36 and 6×26 (the equal-size partners of 12×18 that Q2 needs), plus the two best new seed-42 cells
  by full-val peak PSNR if they differ. Up to 4 runs, about 8 GPU-hours.
- GPUs 0–3 (allowed by the user for this experiment). Workers wait for 44 GB free on their card. Crashed runs rerun in
  full.

## Scoring
As in `brain_varnet_scaling`: `best_model.pt` and `last_model.pt` on the full brain val set (40 volumes, 605 slices)
and on the training slices, with mask S, the common per-slice Bernoulli draws (seeds 1234, 4321), the trainer's
apodized metric mask, per-slice pairing and 95% bootstrap CIs over volumes. The peak checkpoint is still chosen on the
trainer's 10-volume subset (harder than train; see the scaling report's data audit).

Readouts: PSNR / SSIM / NMSE per cell; PSNR against parameters with the equal-size groups marked; PSNR against
wall-clock; train − val gap; val curves.

## Expected outcome
- 80 volumes is still data-limited, so width plateaus like depth did: 18 → 26 or 36 channels adds ≤ +0.1 dB; 12
  channels loses 0.2–0.4 dB.
- At equal size, more cascades beat wider ones (each cascade adds a data-consistency step): 3×36 trails 6×26 ≈ 12×18
  by 0.1–0.3 dB.

## Criteria (full-val PSNR at the peak checkpoint; "both seeds" = paired per slice in s42 and s43)
- **Validity:** every run finishes, brain SSIM ≥ 0.80, no NaNs or skipped steps.
- **Q1, width:**
  - wider helps: the best 26- or 36-channel cell beats the 18-channel cell with the same cascades by ≥ 0.15 dB in both
    seeds (the seed-43 pick rule makes the best new cells two-seed);
  - width saturated: |Δ| < 0.05 dB;
  - 18 needed: 12 channels loses ≥ 0.15 dB to 18 at the same cascades (seed 42, 6 and 12 cascades);
  - otherwise inconclusive.
- **Q2, depth vs width at about 30 M:**
  - depth wins: 6×26 and 12×18 both beat 3×36 by ≥ 0.15 dB in both seeds;
  - width wins: 3×36 beats both by ≥ 0.15 dB in both seeds;
  - size alone: all pairwise |Δ| < 0.05 dB;
  - otherwise inconclusive.
- **Exploratory:** is quality set by parameter count alone? Cells on one diagonal within 0.1 dB of each other.
- **Next step:** the cheapest cell by wall-clock within 0.05 dB of the best becomes the reconstructor for the static vs
  learned mask rerun (S, M, U).
