# Which recon lr and schedule should joint VarNet + LOUPE (R=8) use? (2026-11-14)

| | |
|---|---|
| Code | results produced at `7d46cd6` (varnet_loupe_lr_schedule_r8 masks: right-band and shape metrics, mask snapshots over training); branch `exp/varnet_loupe_lr_schedule_r8` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 49 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 16 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |
| Training commit | not recorded for these runs (they predate `run_info.json`) |

- **Training commit, corrected.** The runs do have a `run_info.json`, but its `commit` is empty because the CoreHPC code copy
  had no `.git`. The first sweep ran from run state `666f872`, the follow-up from `15680ca`, follow-up 2 from `936f3f6`. All
  are on this branch.
- **Branch**: `exp/varnet_loupe_lr_schedule_r8` @ `2e6e37f`. Results at `7d46cd6` (the eval script is unchanged since `9264ce0`, which scored the first two sweeps).
  Return to the code with: `git worktree add ../SeqMRI-worktrees/exp-varnet_loupe_lr_schedule_r8 exp/varnet_loupe_lr_schedule_r8`
- **Proposal vs outcome**: see the three tables below (`proposal.md`, `followup_proposal.md`, `followup2_proposal.md`, all frozen).
- **Status: parked** (user verdict 2026-11-17). The gain is real but mild (+0.12 to +0.17 dB), the grid stops at 4e-3 while
  the gain is still rising, and full local v1 (160 training volumes) is not the full fastMRI train set. Main keeps
  `training.lr=1e-3`. A larger lr grid (8e-3, 1e-2 and beyond) on more data is stashed as a future experiment (Next steps 1).
- **Recommendation at writing**: merge, adopt `lr4e3_step` (recon lr 4e-3, step at 90) as the candidate default. It also wins at
  6 cascades, where it matches the 12-cascade `lr4e3_step` with half the parameters.
- **Stashed proposal**: `next_mask_commitment_proposal.md` asks why the 12-cascade 1e-3 LOUPE masks drop the right band
  (Results 6) and which fix keeps it (logit clamp, slope anneal, random-mask batches). It is not approved and was not run.

### Proposal vs outcome: first sweep (`proposal.md`, 20% data, 400 epochs, 3 seeds, `last_model.pt`)

| Criterion | Outcome | Met |
|---|---|---|
| Adopt: mean ≥ +0.10 dB, 3/3 seeds, no condition < −0.05, SSIM ≥ −0.002; largest mean wins | `lr1e3_const` +0.249 (3/3, conditions +0.25/+0.26/+0.25, SSIM +0.011). `lr2e3_step` +0.128 also passes | **met, `lr1e3_const`**. This is an artifact of overfitting (Results 2). |
| Mask lr: \|`shared_2e3_step` − `lr2e3_step`\| ≥ 0.10 dB, same sign in 3/3 seeds | −0.040 (seeds −0.147, +0.035, −0.006) | not met: no effect |

### Proposal vs outcome: follow-up (`followup_proposal.md`, full data, 100 epochs, 2 seeds, `last_model.pt`)

| Criterion | Outcome | Met |
|---|---|---|
| Adopt: mean ≥ +0.10 dB, 2/2 seeds, no condition < −0.05, SSIM ≥ −0.002; largest mean wins | `lr4e3_step` +0.171 (+0.174, +0.169; brain +0.111, knee-PD +0.273, knee-PDFS +0.123; SSIM +0.0006). `lr2e3_step` +0.123 also passes | **met, `lr4e3_step`** |
| Default flip needs a third seed of winner and control | not run | pending |
| Cosine: `lr4e3_wcos` − `lr4e3_step`, ≥ 0.10 dB, same sign | −0.171 (−0.068, −0.273) | met: step better |
| Cosine: `lr2e3_wcos` − `lr2e3_step` | −0.212 (−0.026, −0.398) | met: step better |
| Overfitting flag: val peaks before epoch 80 | brain 9/10 runs, knee 8/10 | **tripped**: brain drops 0.01–0.30 dB from peak, knee 0.02–0.11 dB (Results 4) |

### Proposal vs outcome: follow-up 2 (`followup2_proposal.md`, 6 cascades, full data, 100 epochs, 2 seeds, `last_model.pt`)

| Criterion | Outcome | Met |
|---|---|---|
| Adopt at 6 cascades: mean ≥ +0.10 dB, 2/2 seeds, no condition < −0.05, SSIM ≥ −0.002; largest mean wins | `lr4e3_step` +0.117 (+0.113, +0.121; brain +0.117, knee-PD +0.180, knee-PDFS +0.052; SSIM +0.0012). `lr2e3_step` +0.059 fails | **met, `lr4e3_step`** |
| Interaction `lr4e3_step`: shrinks if d6 − d12 ≤ −0.08 with both seeds lower; holds if \|d6 − d12\| < 0.08 | −0.055 (d6 +0.113/+0.121 vs d12 +0.174/+0.169) | **holds**, though lower in both seeds |
| Interaction `lr2e3_step` | −0.064 (d6 +0.034/+0.084 vs d12 +0.085/+0.160) | **holds**, though lower in both seeds |
| Cascade count (reported): 6-cascade minus 12-cascade control | +0.054 pooled (+0.066, +0.042); knee-PDFS +0.116 | reported |

## Question

The recon optimizer (Muon + LayerNorm, main's default) ran at lr 1e-3 with a ×0.1 step 10% before the end. Is a higher lr or
another schedule better for joint LOUPE + VarNet at R=8? A second question was whether the LOUPE mask logits should share the recon
lr. Normally they keep their own Adam lr, pinned at 1e-3. `training.lr` sets the whole recon: the Muon group (the U-Nets' hidden
convolutions; `muon_lr` is unset, so it falls back to `training.lr`) and the Adam group (everything else in the VarNet and the
sensitivity net). All differences are paired: arm minus the `lr1e3_step` control of the
same training seed, in dB of PSNR_Masked, scored on the same val slices with the same mask draws.

## Runs compared

Common to all runs: `--model sequential recon=varnet datasets=fastmri_local_v1_flair_knee misc.resolution=[320,320]
sampler.accelerations=[8.0] sampler.budget_over=acquired sampler.sampler_type=loupe_2d sampler.loupe_vd_init_power=0
sampler.anatomy_conditioning=false training.num_step=1 training.batch_size=16`. The mask lr is `sampler.lr=0.001`, except in
`shared_2e3_step`. The schedule scales every group, the mask included. All runs trained on CoreHPC, one RTX Pro 6000 each.

| Group | Data, epochs | Arms (recon lr, schedule) | Seeds | Control, full val, `last_model` (pooled PSNR / SSIM) |
|---|---|---|---|---|
| First sweep, `checkpoints/vlls_*` | `limit_volumes=0.2` (32 train volumes, 772 slices), 400 ep, val every 10 | {5e-4, 1e-3, 2e-3} × {step at 360, warmup_cosine (10 ep warmup), wsd (10 ep warmup, linear decay over the last 80 ep)}; `lr1e3_const`; `shared_2e3_step` (mask lr 2e-3 too) | 42, 43, 44 | 30.58–30.71 dB / 0.777–0.780 (`best_model`: 32.66–32.78 / 0.850–0.852) |
| Follow-up, `checkpoints/vlls_full_*` | `limit_volumes=1` (tier S, 160 volumes), 100 ep, val every 5 | `lr1e3_step` (control), `lr2e3_step`, `lr4e3_step` (step at 90), `lr2e3_wcos`, `lr4e3_wcos` (2 ep warmup) | 42, 43 | 33.77–33.81 dB / 0.862–0.863 |
| Follow-up 2, `checkpoints/vlls_full_c6_*` | as the follow-up, `recon.num_cascades=6` (15.3M parameters, against 30.1M) | `lr1e3_step` (control), `lr2e3_step`, `lr4e3_step` | 42, 43 | 33.81–33.88 dB / 0.862–0.863 |

warmup_cosine ramps from 10% to 100% of the lr, then follows a cosine down to a 1% floor. wsd ramps up, holds, then decays linearly.
The follow-up control matches the earlier 100-epoch Muon + LayerNorm runs, `reports/muon_layernorm_100ep/report.md` (about 33.7 dB pooled, SSIM 0.86).

## Methods

1. **Full-val scoring.** `scripts/eval/varnet_loupe_lr_schedule_r8_eval.py` loads each finished run's checkpoint. It scores the
   checkpoint on the full local v1 val set: brain 40 volumes / 605 slices, knee-PD 20 / 673, knee-PDFS 20 / 660. Scoring uses the
   trainer's metric mask, each slice's own data range, and 2 seeded mask draws (1234, 4321), via
   `varnet_train_speed_eval.score`. The first sweep's 20% val is a subset of this set.
2. **Paired difference.** d(arm, seed) is the mean over slices of arm minus control at the same training seed. The 95% CI is
   ±1.96 standard errors over the 80 per-volume mean differences, so it covers val volumes, not training noise.
3. **Criteria and twins.** `scripts/eval/varnet_loupe_lr_schedule_r8_summary.py` applies the frozen rules and writes
   `summary.json`. A twin contrast (A − B) is d(A) − d(B) per seed.
4. **Val curves.** The same script reads each run's trainer val PSNR per anatomy from `train_sequential.log` and reports its peak
   epoch and the drop from peak to the last epoch. The trainer's per-anatomy PSNR uses the last batch's data range, so it is on a
   different scale from Methods 1. It is used only for timing.
5. **Cascade interaction** (follow-up 2). d6 and d12 are the paired gains of an arm against the control of its own cascade
   count, per seed. d6 − d12 is their seed-mean difference. The 6- vs 12-cascade control comparison is unpaired: the
   difference of absolute pooled PSNR at the same seed.
6. **LOUPE masks** (post hoc, `scripts/plots/varnet_loupe_lr_schedule_r8_masks.py`, CPU). For each run it reads the
   320x320 mask logits from `last_model.pt` and `best_model.pt` and works on p = sigmoid(10 · logits), before the per-slice
   budget rescale. Per run: support (share of the grid with p > 0.05), share of total p in radius bands, and the
   **right band**, the total p in rows within 12 of the centre row and columns more than 72 right of the centre column.
   Per pair of masks: Pearson correlation of p, the same after a 9x9 box filter (shape), and the Jaccard overlap of the
   supports. The only record of the masks during training is the trainer's val `prob_mask` image logged to wandb every 10
   epochs (rendered figures with a colour scale per image), tiled in `figures/masks_over_training.png`.
7. **Post hoc** (not in any proposal): the first sweep scored at `best_model.pt`, the checkpoint that peaked on the trainer's
   20% val.

## Results

### 1. Follow-up: full data, 100 epochs (`eval_runs_full_last_model.json`)

| Arm | d, s42 [95% CI] | d, s43 [95% CI] | Mean | brain | knee-PD | knee-PDFS | SSIM |
|---|---|---|---|---|---|---|---|
| `lr4e3_step` | +0.174 [+0.13, +0.22] | +0.169 [+0.14, +0.20] | **+0.171** | +0.111 | +0.273 | +0.123 | +0.0006 |
| `lr2e3_step` | +0.085 [+0.06, +0.11] | +0.160 [+0.14, +0.18] | +0.123 | +0.104 | +0.182 | +0.079 | +0.0007 |
| `lr4e3_wcos` | +0.106 [+0.07, +0.14] | −0.105 [−0.17, −0.04] | +0.001 | −0.137 | +0.014 | +0.113 | −0.0007 |
| `lr2e3_wcos` | +0.059 [+0.03, +0.09] | −0.239 [−0.31, −0.17] | −0.090 | −0.175 | −0.169 | +0.069 | −0.0017 |

A higher lr with the step schedule gains in every condition, and the gain grows from 2e-3 to 4e-3. Nothing diverged at 4e-3.

| Twin | s42 | s43 | Mean | brain | knee-PD | knee-PDFS |
|---|---|---|---|---|---|---|
| `lr4e3_wcos` − `lr4e3_step` | −0.068 | −0.273 | −0.171 | −0.248 | −0.258 | −0.011 |
| `lr2e3_wcos` − `lr2e3_step` | −0.026 | −0.398 | −0.212 | −0.279 | −0.351 | −0.010 |

Cosine loses to step at the same peak lr on brain and knee-PD, and ties on knee-PDFS. The frozen bar is met, but seed 42 alone
falls short of 0.10 dB in both twins.

### 2. First sweep: 20% data, 400 epochs (`eval_runs_last_model.json`, `eval_runs_best_model.json`)

| Arm | `last_model`: mean (s42, s43, s44) | `last_model` SSIM | `best_model` (post hoc): mean (s42, s43, s44) |
|---|---|---|---|
| `lr1e3_const` | **+0.249** (+0.248, +0.267, +0.233) | +0.0109 | −0.021 (−0.006, −0.087, +0.030) |
| `lr2e3_step` | +0.128 (+0.150, +0.134, +0.098) | +0.0018 | **+0.128** (+0.107, +0.170, +0.106) |
| `shared_2e3_step` | +0.088 (+0.003, +0.169, +0.092) | +0.0031 | −0.106 (−0.111, +0.027, −0.234) |
| `lr2e3_wcos` | −0.055 (+0.090, −0.337, +0.081) | +0.0020 | −0.157 |
| `lr2e3_wsd` | −0.072 (−0.062, −0.184, +0.031) | −0.0017 | −0.042 |
| `lr1e3_wsd` | −0.188 (−0.097, −0.357, −0.111) | −0.0050 | −0.126 |
| `lr1e3_wcos` | −0.243 (−0.077, −0.518, −0.135) | −0.0021 | −0.259 |
| `lr5e4_step` | −0.254 (−0.120, −0.570, −0.074) | −0.0027 | −0.683 |
| `lr5e4_wsd` | −0.368 | −0.0080 | −0.416 |
| `lr5e4_wcos` | −0.440 | −0.0059 | −0.622 |

Every run peaks on the trainer val at epoch 50–110 and then loses 1.6–2.4 dB by epoch 400. `last_model` therefore measures how
slowly an arm overfits. `lr1e3_const` loses less than the control in all 3 seeds (1.90–1.94 dB on brain, against 2.07–2.13), so it wins at
`last_model` and ties at the peak. `lr2e3_step` is the only arm positive at both checkpoints, in 3/3 seeds each.

| Twin, first sweep | s42 | s43 | s44 | Mean |
|---|---|---|---|---|
| `shared_2e3_step` − `lr2e3_step`, `last_model` (frozen) | −0.147 | +0.035 | −0.006 | −0.040 |
| same, `best_model` (post hoc) | −0.219 | −0.143 | −0.341 | −0.234 |

The frozen mask-lr contrast is null. At the peak, sharing the 2e-3 lr with the mask costs 0.23 dB in 3/3 seeds, but that
checkpoint was chosen on part of the scored val.

### 3. Seed 43

In both sweeps, seed 43 is the low seed of every cosine, wsd and 5e-4 arm; the step arms at lr ≥ 2e-3, `lr1e3_const` and `shared_2e3_step` do not show it. Examples: −0.337
(`lr2e3_wcos`), −0.570 (`lr5e4_step`) and −1.57 (`lr5e4_step`, `best_model`) in the first sweep, and −0.273 / −0.398 in the
follow-up cosine twins. In those arms seed 43 sits 0.1–1.3 dB below seeds 42 and 44.

### 4. Overfitting at full data (trainer val, `summary.json` → `followup.curves`)

| Run | brain peak epoch | brain drop to ep 100 (dB) | knee peak epoch | knee drop (dB) |
|---|---|---|---|---|
| `lr1e3_step` s42 / s43 | 95 / 60 | 0.01 / 0.14 | 65 / 45 | 0.05 / 0.11 |
| `lr2e3_step` s42 / s43 | 70 / 65 | 0.08 / 0.05 | 70 / 80 | 0.05 / 0.04 |
| `lr4e3_step` s42 / s43 | 70 / 65 | 0.09 / 0.17 | 70 / 75 | 0.05 / 0.04 |
| `lr2e3_wcos` s42 / s43 | 65 / 55 | 0.08 / 0.18 | 75 / 85 | 0.02 / 0.02 |
| `lr4e3_wcos` s42 / s43 | 60 / 40 | 0.06 / 0.30 | 75 / 45 | 0.02 / 0.03 |

Knee is flat from about epoch 70: drops of 0.02–0.11 dB are within the trainer val's epoch-to-epoch noise. Brain overfits a
little, and most in the cosine arms on seed 43. The flag trips by the frozen wording, but its size is 0.1–0.3 dB on brain, not
the 2 dB of the first sweep.

### 5. Follow-up 2: 6 cascades (`eval_runs_full_c6_last_model.json`, `summary.json` → `followup2`)

| Arm | d6, s42 [95% CI] | d6, s43 [95% CI] | Mean | brain | knee-PD | knee-PDFS | SSIM | d12 mean | d6 − d12 |
|---|---|---|---|---|---|---|---|---|---|
| `lr4e3_step` | +0.113 [+0.09, +0.13] | +0.121 [+0.10, +0.14] | **+0.117** | +0.117 | +0.180 | +0.052 | +0.0012 | +0.171 | −0.055 |
| `lr2e3_step` | +0.034 [+0.02, +0.05] | +0.084 [+0.07, +0.09] | +0.059 | +0.044 | +0.096 | +0.034 | +0.0005 | +0.123 | −0.064 |

`lr4e3_step` is better than the 6-cascade control on 77/80 and 78/80 val volumes. The gain is about a third smaller than at 12
cascades, in both seeds and both arms, but the drop is below the frozen 0.08 dB bar.

Absolute pooled PSNR (full val, `last_model`), s42 / s43:

| Arm | 12 cascades | 6 cascades | 6 − 12 |
|---|---|---|---|
| `lr1e3_step` | 33.810 / 33.767 | 33.876 / 33.809 | +0.066 / +0.042 |
| `lr2e3_step` | 33.896 / 33.927 | 33.910 / 33.893 | +0.014 / −0.034 |
| `lr4e3_step` | 33.985 / 33.936 | 33.989 / 33.930 | +0.004 / −0.006 |

At 1e-3, 6 cascades beat 12 by +0.054 dB (brain +0.026, knee-PD +0.017, knee-PDFS +0.116 in both seeds). At 4e-3 the two
depths tie to within 0.006 dB. The lr gain shrinks at 6 cascades because the 6-cascade control is better, not because 6-cascade
`lr4e3_step` is worse. The 6-cascade runs do not overfit brain: trainer val peaks at epoch 80–95 with drops ≤ 0.02 dB (12
cascades: peaks 60–95, drops 0.01–0.17). Training takes about 87 s per epoch at 6 cascades on CoreHPC.

### 6. LOUPE masks (`masks.json`, `figures/masks.png`, `figures/masks_over_training.png`; post hoc)

The masks were learned jointly in every run, with the same Adam lr (1e-3) and the same schedule for the logits in every arm.
Any difference between arms therefore comes through the recon's gradients.

All masks share one shape: a dense centre and a lopsided footprint covering about 22% of the grid. The left half is sampled
out to the edge, while the right half stops about 70 columns from the centre. Correlations after smoothing are 0.89–0.999
between any two full-data masks. Pixel-level speckle is set by the start mask's seed: unsmoothed p correlates 0.80–0.89
between arms of the same seed, against 0.50–0.60 between seeds of the same arm.

The one clear difference is the **right band**: a strip along the centre row, from 72 columns right of centre to the edge of
k-space.

| Run (full data, `last_model`) | Right band, s42 / s43 (sum of p) | Paired gain vs own-depth control, s42 / s43 (dB) |
|---|---|---|
| 12c `lr1e3_step` (control) | **8 / 0** | — |
| 12c `lr2e3_step` | **25** / 454 | +0.085 / +0.160 |
| 12c `lr4e3_step` | 396 / 475 | +0.174 / +0.169 |
| 12c `lr2e3_wcos`, `lr4e3_wcos` | 413 / 248, 455 / 429 | +0.059 / −0.239, +0.106 / −0.105 |
| 6c `lr1e3_step` (control) | 451 / 469 | — |
| 6c `lr2e3_step` | 403 / 462 | +0.034 / +0.084 |
| 6c `lr4e3_step` | 463 / 459 | +0.113 / +0.121 |

- **Only the 12-cascade 1e-3 runs leave the band empty**, in both seeds; so does 12c `lr2e3_step` s42. That run has the
  smaller of the two 2e-3 gains (+0.085 against +0.160). Every other full-data run samples it, including both 6-cascade
  controls.
- **The band is decided early.** In the wandb snapshots, 4e-3 and the 6-cascade control show it by epoch 10–20, and the
  12-cascade 1e-3 control never grows it. Masks barely move after epoch 60: best vs last checkpoint correlate ≥ 0.98.
- **The band does not explain everything.** The cosine arms have it and still lose or tie. At 6 cascades, where every arm has
  it, 4e-3 still gains +0.117.
- **The shrinking 4e-3 gain fits the band.** At 12 cascades, the control's missing band is one more thing 4e-3 fixes;
  at 6 cascades, the control already has it. d6 − d12 = −0.055 dB is of the same order as the 6c − 12c control gap (+0.054).
  Hypothesis, untested: no run swaps or freezes the mask.
- **20% data:** the band is present in 2 of 3 control seeds (485, 659, 143), so the empty band is specific to the 12-cascade
  full-data control. The 20% masks keep binarizing after the peak (mean |2p − 1| 0.85 at `best_model`, 0.95 at epoch 400)
  and shrink their support from about 0.30 to 0.15.

**Earlier lr evidence.** The only earlier VarNet lr study is the 2026-10-19/20 sweep (`reports/lr_sweep_r8/paired.json`;
no write-up). It used Adam, a frozen variable-density mask (no LOUPE), the pre-local-v1 data, one seed and 50 epochs, so it
is not a control here. At 1e-3 step vs 3e-3 warmup_cosine it found a tie (brain −0.04, knee +0.05 dB), and 1e-3 cosine lost
(−0.26 / −0.34). A higher lr with step decay was never run. That matches this study, where 4e-3 cosine also ties the 1e-3
step control (+0.001). It does not show that 1e-3 is best for a frozen mask, and it says nothing about Muon. Adam and Muon
2e-3 runs were queued in `reports/varnet_train_speed/report.md` but never ran.

## Conclusions

- **At full data and 100 epochs, recon lr 4e-3 with step at 90 beats main's 1e-3** by +0.171 dB pooled. Both seeds gain
  +0.17, every condition is positive, and SSIM is unchanged (Results 1). 2e-3 gains +0.123, so the gain is still rising at 4e-3, the
  top of the grid.
- **Cosine is worse than step at the same peak lr** at both 2e-3 and 4e-3. The loss is 0.17–0.21 dB pooled, on brain and knee-PD.
  It meets the frozen bar but depends mostly on seed 43 (Results 1, 3).
- **A higher peak lr did not rescue cosine.** `lr4e3_wcos` ties the 1e-3 step control (+0.001).
- **The first sweep's regime overfits and does not answer the lr question.** 32 training volumes peak by about epoch 70 and lose
  about 2 dB by epoch 400. Its frozen winner, `lr1e3_const`, overfits more slowly but does not optimize better: at the peak it ties
  the control. Its one robust result agrees with the follow-up: 2e-3 step beats 1e-3 step at both checkpoints (+0.128).
- **Mask lr: no effect at the frozen checkpoint.** Sharing a 2e-3 lr with the mask logits is −0.040 dB. The post hoc
  `best_model` contrast (−0.234, 3/3 seeds) suggests that a larger mask step hurts. Hypothesis, untested at full data.
- **The 4e-3 gain holds at 6 cascades by the frozen rule** (+0.117 dB, d6 − d12 = −0.055). Part of the 12-cascade gain
  looks like making up for excess depth: at 1e-3, 6 cascades beat 12 by +0.05 dB, and at 4e-3 they tie. The rest (+0.12 dB)
  remains at a depth that does not overfit, so it is not only regularization of an oversized network (Results 5).
- **6 cascades are enough at this data size** with 4e-3: 33.99 / 33.93 dB, the same as 12 cascades, at half the recon cost.
- **The 12-cascade 1e-3 control is the only full-data setup whose LOUPE mask never samples the right band**, a strip of
  high-frequency columns along the centre row (Results 6). Every 4e-3 run and both 6-cascade controls sample it, and the
  12-cascade 2e-3 run without it has the smaller 2e-3 gain. So part of the 12-cascade lr gain may come through the mask.
  Hypothesis, untested: it is a correlation over 2 seeds, and a 4e-3 gain of +0.12 dB remains at 6 cascades, where every
  arm has the band.
- **Lower lr loses everywhere.** 5e-4 is −0.25 to −0.44 dB in the first sweep at every schedule, so main's 1e-3 sits on the
  low side.

## Caveats

- **The first sweep's `last_model` SSIM is below 0.80.** Pooled 0.777–0.780, brain 0.794–0.796, knee-PDFS 0.68–0.69. The same runs
  through the same eval path give 0.850–0.852 at their `best_model`, and the full-data runs give 0.862–0.863. So this is the
  2 dB overfitting loss, not a scoring bug.
- **Knee-PDFS SSIM is below 0.80 in every full-data run** (0.783–0.784 for the control). This predates the study
  (`reports/muon_layernorm_100ep/report.md`) and is still undebugged. The comparisons are paired.
- **Two seeds in the follow-up**, and seed 43 drives both cosine verdicts. The CIs cover val volumes only. The earlier
  run-to-run spread of joint LOUPE + VarNet training is 0.3–0.5 dB. The `lr4e3_step` gain is unusually stable (+0.174 and
  +0.169), but its third seed has not been run.
- **Brain overfits slightly in the 100-epoch runs** (Results 4), so a little of each difference is overfitting tolerance. It
  explains only part of cosine's loss. On the trainer's scale, seed 43's cosine runs are already below step at their brain
  peaks: `lr4e3_wcos` 37.59 vs `lr4e3_step` 37.82 dB, and `lr2e3_wcos` 37.45 vs `lr2e3_step` 37.74. On seed 42 the peaks are within
  0.03–0.07 dB. Their larger drop after the peak (0.18–0.30 vs 0.05–0.17 dB) adds about 0.1 dB on top.
- **Seed 43 mechanism.** Why seed 43 hurts the non-step arms is unknown. Candidates are the LOUPE mask trajectory and the data
  order. Hypothesis, untested.
- **Post-hoc results.** The `best_model` scores, the seed-43 pattern and the curve-peak table were not in either proposal.
  `best_model` was chosen on the 20% val, which is part of the scored val.
- **Follow-up 2 is two seeds.** Both interactions fall between "holds" and "shrinks" (−0.055, −0.064) and are lower in both
  seeds; a third seed could move them across the 0.08 bar. The 6- vs 12-cascade comparison is unpaired and its +0.05 dB is
  inside the usual seed spread.
- **Mask analysis is post hoc and descriptive.** The right band was found by looking at the maps; its region (±12 rows,
  > 72 columns) was set after looking. It uses raw p before the per-slice budget rescale, and no run tests it causally.
- **Grid edge.** 4e-3 is the largest lr tried; the optimum may be higher.
- **Scoring location.** The runs trained on CoreHPC and were scored here from an rsync copy, against the local copy of local v1
  (`--data-root`). The val slice counts match the training logs (605 / 673 / 660).
- **wandb**: the runs logged offline on CoreHPC and were synced on 2026-11-14 and 2026-11-17: 49 runs named `vlls_*` in project SeqMRI. Their
  wandb group came out as `runs` / `runs_full` / `runs_full_c6`, not the `WANDB_RUN_GROUP` the sbatch set.

## Figures

- `figures/masks.png`: final masks of the 1e-3 and 4e-3 step arms at 12 and 6 cascades, both seeds, full 320x320 grid;
  radial profiles; and the 4e-3 minus 1e-3 difference for seed 42.
- `figures/masks_over_training.png`: the trainer's val prob_mask images every 10 epochs for six full-data runs (12c 1e-3
  and 4e-3, both seeds; 6c 1e-3 and 4e-3, s42).

## Next steps

1. **Where does the lr gain stop, on more data?** `lr8e3_step` and `lr1e2_step` (plus 4e-3 and the 1e-3 control), 2 seeds, at
   6 cascades, on a larger train set than local v1 (e.g. tier F FLAIR plus more knee). Watch for NaNs; PromptMR diverged at 1e-3,
   but the VarNet U-Net with Muon has not up to 4e-3. Add a third seed of 4e-3 before changing main's default.
2. **LOUPE mask commitment** (`next_mask_commitment_proposal.md`): does a logit clamp, a slope anneal or random-mask batches keep
   the right band in the 12-cascade 1e-3 control and close its gap to 4e-3? This separates sigmoid saturation from an early race
   between mask and recon.
3. **Mask lr at full data**: `shared_4e3_step` vs `lr4e3_step`, to test the post hoc −0.23 dB.
4. **Early stop or a shorter schedule**: brain peaks near epoch 65 at 4e-3, so an earlier decay may lose nothing on brain and
   save time.

## Reproduce

`bash reports/varnet_loupe_lr_schedule_r8/run.sh`. The mask comparison needs only a CPU. It needs the 49 runs at
`/export/scratch/bkadota/varnet_loupe_lr_schedule_r8/{runs,runs_full,runs_full_c6}`, local v1 at `/export/scratch/bkadota/fastmri_raw`, and
one GPU (about 1 h per first-sweep eval without the per-slice caches). Check with
`python scripts/report_manifest.py check reports/varnet_loupe_lr_schedule_r8`.
