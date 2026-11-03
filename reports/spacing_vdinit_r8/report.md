# Does sample spacing, or a variable-density start, help LOUPE + VarNet at R=8? (2026-10-23)

> **Provenance (written by hand, not by `scripts/report_manifest.py`).** The runs live in `~/scratch/code/loupe_spacing_vdinit_r8/runs/`, not in
> `checkpoints/`, so `report_manifest.py write` cannot index them and there is no `manifest.json`.
> - **Code:** snapshot of the local SeqMRI tree at HEAD `a7e576b` plus uncommitted changes (`code/SNAPSHOT.txt` in the experiment folder), taken 2026-10-19 23:00. The new pieces are `sampler.fixed_mask_path` (`activemri/samplers/fixed_mask.py`) and `sampler.loupe_vd_init_power`.
> - **Jobs:** SLURM array `2348509`, tasks 0-8, all `COMPLETED` (exit 0), first wandb run started 2026-10-20 16:27, last (`topk_loupe`) started 2026-10-21 00:45. Smoke tests: jobs `2345389`, `2347024` (`runs/debug/`).
> - **Metric mask: legacy** (`target > 0.1 * max`). The snapshot predates the apodized mask (2026-10-20), and no `metric_mask.py` is in it. Do not compare these numbers with apodized-mask reports.
> - **Data:** balanced brain T2 + knee split; train 89 + 40 volumes (1,350 slices per anatomy), val 23 + 10 volumes (344 slices per anatomy, `n=344` in every log).
> - **Results:** `summary.md`, `summary.json` (parsed from `train_sequential.log`, last epoch), `masks_r8.json`, `figures/masks_r8.png`. Regenerate with `bash reports/spacing_vdinit_r8/run.sh`.

## Question

Two questions, run as one array of nine R=8 trainings (VarNet, brain T2 + knee):
1. **Spacing.** The learned LOUPE mask's pixel spacing looks like an independent random draw (nearest-neighbour pair correlation 0.86 vs 0.87). Does a Poisson-disc mask with the same density, centre and sample count beat an independent draw? If it does, position-aware mask learners (BASS, AutoSamp) have room over LOUPE.
2. **VD-init.** Does starting LOUPE from a variable-density prior (Sun et al., TCI 2024), `clamp(c((D-r)/D)^4, 0.01, 0.99)` with `c` set to the sample budget, beat the default uniform-random start?

Metrics are PSNR (dB) and SSIM on the validation set, per anatomy, at the final epoch (50), under the trainer's background-masked metric. Acceleration 8 means 8,192 of 65,536 positions (Sparsity 0.125), with a fixed 10 x 10 centre.

## Runs compared

Recipe shared by all arms: `--model loupe --recon varnet --dataset brain_knee_balanced --acceleration 8`, batch 16, 50 epochs, lr 1e-3 with a step at epoch 40, L1 loss, slope 5, mask and STE grad floors 0.05. Run directories: `runs/<arm>-2348509/` in the experiment folder.

| task | arm | what trains | mask | brain PSNR / SSIM | knee PSNR / SSIM | wall time |
|---|---|---|---|---|---|---|
| 0 | `random_s1` | LOUPE + VarNet, seed 1 | learned, uniform-random start | 36.60 / 0.9325 | 34.26 / 0.9543 | 1h41 |
| 1 | `random_s2` | same, seed 2 | same | 36.53 / 0.9318 | 34.29 / 0.9545 | 3h19 |
| 2 | `vdinit_s1` | LOUPE + VarNet, seed 1 | learned, VD-prior start | 36.42 / 0.9315 | 34.10 / 0.9528 | 1h46 |
| 3 | `vdinit_s2` | same, seed 2 | same | 36.24 / 0.9304 | 33.99 / 0.9517 | 3h33 |
| 4 | `pd_loupe_s0` | VarNet only, seed 42 | Poisson-disc draw of the learned density, mask seed 0 | 36.72 / 0.9333 | 34.34 / 0.9555 | 1h34 |
| 5 | `pd_loupe_s1` | same | mask seed 1 | 36.70 / 0.9331 | 34.28 / 0.9552 | 3h15 |
| 6 | `white_loupe_s0` | VarNet only, seed 42 | independent draw of the same density, mask seed 0 | 36.53 / 0.9318 | 34.25 / 0.9543 | 1h34 |
| 7 | `white_loupe_s1` | same | mask seed 1 | 36.58 / 0.9322 | 34.28 / 0.9545 | 3h16 |
| 8 | `topk_loupe` | VarNet only, seed 42 | the 8,092 highest-probability locations, deterministic | 36.66 / 0.9331 | 34.23 / 0.9541 | 1h34 |

All nine finished 50/50 epochs and every SSIM is about 0.93 (brain) and 0.95 (knee), so the models are healthy. The local reference run that supplied the density (`checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05`, seed 42) scored brain 36.55 / 0.9328 and knee 34.26 / 0.9529, which sits inside the random-init range here (the HPC environment differs: torch 2.12 vs 2.6, different GPUs). Wall time varies by 2x for identical work because of GPU sharing and node differences, so it carries no information about the arms.

In the fixed-mask arms the LOUPE sampler is a placeholder: the saved mask replaces it in train and val, and only VarNet (with its sensitivity net) trains.

## Terms

All three fixed-mask families start from the same learned LOUPE density p(k) (the probability of sampling each k-space location) and the same fixed 10 x 10 centre, and hold 8,192 samples in total.
- **White draw** (`white_loupe_*`): locations sampled independently of one another, so neighbours can clump or both be missed. "White" is from white noise, which has no correlation between points. Made by drawing uniform noise `u` per location and keeping the 8,092 non-centre locations with the smallest `u/p`, which gives an exact count. Its pair correlation at 1 px is about 1.
- **Poisson-disc** (`pd_loupe_*`): same density, but samples actively avoid landing next to each other (pair correlation at 1 px well below 1). PD vs white therefore isolates the effect of spacing.
- **Top-k** (`topk_loupe`): no randomness. Take the 8,092 non-centre locations with the highest p(k) and sample exactly those. It gives a dense, coherent core with sparse outer pixels, and moves samples inward relative to the learned density (ring error 31% mean). It tests whether making the mask deterministic helps or hurts.

## Methods

1. **Mask construction** (`scripts/diagnostics/make_spacing_masks_r8.py`, run locally against the reference checkpoint). All five masks share the learned density, the 10 x 10 centre and exactly 8,192 samples.
   - `pd_loupe_*`: variable-density Poisson-disc, with the input density iteratively compensated until each ring's count matches the learned density (max ring error 5.9% and 3.4%, mean 1.6% and 1.0%). A plain Poisson-disc draw under-samples inner rings by up to 57%, which would confound spacing with density.
   - `white_loupe_*`: the 8,092 non-centre locations with the smallest `u/p`, with `u` white noise. An independent draw at an exact count (max ring error 5.9% and 12.5%).
   - `topk_loupe`: the 8,092 highest-probability locations (ring error 98% max, 31% mean; it moves samples inward).
   - Pair correlation `g` at 1 px (1 = independent, below 1 = neighbours avoided), by k-space radius 0-24 / 24-48 / 48-96 / 96+ px: PD 0.94 / 0.59 / 0.30 / 0.12 (seed 0); white 1.06 / 1.05 / 0.96 / 1.07 (seed 0); top-k 1.71 / 1.31 / 0.74 / 0.20. At 2-3 px `g` is about 1 for both PD and white, so spacing acts only between adjacent pixels.
2. **Training** (`train.py`, SLURM array `run_spacing_vdinit.sbatch`, one task per arm). Random and VD-init arms train LOUPE and VarNet jointly with `training.seed` 1 or 2. Fixed-mask arms use `training.seed=42` and `sampler.fixed_mask_path`.
3. **Summary** (`scripts/eval/summarize_spacing_vdinit.py`). Takes the last validation epoch of each log and computes paired differences: VD-init minus random (same seed), PD minus white (same mask seed), top-k minus `white_loupe_s0`. It also tabulates mean val PSNR per epoch.

## Results

**Paired differences, PSNR in dB (positive favours the first arm).**

| comparison | brain dPSNR | knee dPSNR | mean dPSNR | brain dSSIM | knee dSSIM |
|---|---|---|---|---|---|
| vdinit_s1 - random_s1 | -0.180 | -0.160 | -0.170 | -0.0010 | -0.0015 |
| vdinit_s2 - random_s2 | -0.290 | -0.300 | -0.295 | -0.0014 | -0.0028 |
| pd_loupe_s0 - white_loupe_s0 | +0.190 | +0.090 | +0.140 | +0.0015 | +0.0012 |
| pd_loupe_s1 - white_loupe_s1 | +0.120 | +0.000 | +0.060 | +0.0009 | +0.0007 |
| topk_loupe - white_loupe_s0 | +0.130 | -0.020 | +0.055 | +0.0013 | -0.0002 |

VD-init is worse than random init in both seeds and both anatomies (-0.16 to -0.30 dB). Poisson-disc is ahead of the white draw in 3 of 4 anatomy-seed cells, by 0.09-0.19 dB, and tied in the fourth (knee, seed 1).

**Averages over the two seeds** (computed from the table in `summary.md`):

| group | brain PSNR | knee PSNR | mean PSNR |
|---|---|---|---|
| random-init LOUPE (seeds 1, 2) | 36.565 | 34.275 | 35.420 |
| VD-init LOUPE (seeds 1, 2) | 36.330 | 34.045 | 35.188 |
| Poisson-disc (mask seeds 0, 1) | 36.710 | 34.310 | 35.510 |
| white draw (mask seeds 0, 1) | 36.555 | 34.265 | 35.410 |
| top-k (one mask) | 36.660 | 34.230 | 35.445 |

- VD-init minus random-init: brain -0.235, knee -0.230 dB.
- Poisson-disc minus white: brain +0.155, knee +0.045, mean +0.100 dB.
- Random-init joint LOUPE (35.420) and the white draw of its own density (35.410) are within 0.01 dB. Poisson-disc is 0.09 dB above the learned mask itself, but the two use different training seeds (1, 2 vs 42), so that last gap is less clean than the paired one.

**Noise floor.** The two random-init seeds differ by 0.07 dB (brain) and 0.03 dB (knee), mean 0.02 dB. The two white mask seeds differ by 0.05 / 0.03 dB, and the two PD mask seeds by 0.02 / 0.06 dB. So differences below about 0.05 dB are not distinguishable from seed noise.

**Convergence (mean of brain and knee val PSNR, dB).**

| arm | ep 1 | ep 5 | ep 10 | ep 20 | ep 30 | ep 40 | ep 50 |
|---|---|---|---|---|---|---|---|
| random_s1 | 31.02 | 34.33 | 34.91 | 35.21 | 35.31 | 35.38 | 35.43 |
| random_s2 | 31.10 | 34.30 | 34.94 | 35.09 | 35.30 | 35.30 | 35.41 |
| vdinit_s1 | 33.27 | 34.42 | 34.83 | 35.10 | 32.77 | 35.19 | 35.26 |
| vdinit_s2 | 32.76 | 34.21 | 34.58 | 34.90 | 35.05 | 35.00 | 35.12 |

VD-init leads by about 1.7-2.2 dB at epoch 1. By epoch 5 the lead is +0.09 dB in seed 1 and -0.09 dB in seed 2, and from epoch 10 on it trails in both seeds. The `vdinit_s1` value at epoch 30 (32.77) is a one-epoch validation dip, not a collapse: the training loss at that epoch rose from 0.0515 to 0.0522 and validation loss from 0.0538 to 0.0631, and by epoch 31 the PSNR was back to 34.8 (brain 35.77 / knee 33.80). It does not affect the final numbers.

**Top-k.** The deterministic mask did not trail the white draw as expected: brain +0.13, knee -0.02 dB against `white_loupe_s0`, and +0.035 dB against the mean of the two white seeds.

## Conclusions

- **VD-init does not help LOUPE at R=8 here.** It speeds the first epoch or two, then is overtaken and finishes 0.17-0.30 dB below random init in both seeds and both anatomies, roughly 10x the random-seed spread. This does not reproduce the speed-up claimed for the prior at the epoch 5-20 range; it ends below the random start.
- **Spacing gives a small, brain-led gain at the 1-px scale.** Poisson-disc beats an independent draw of the same density by +0.155 dB on brain (both seeds positive, 0.12 and 0.19) and +0.045 dB on knee (0.09 and 0.00). The mean of +0.10 dB sits at the pre-set "worth building" boundary, but the knee half does not meet the pre-set bar (it ties in seed 1). Treat it as suggestive for brain and unresolved for knee.
- **LOUPE's learned positions are equivalent to a random draw of its density.** The jointly trained LOUPE and the white draw of its density land within 0.01 dB, so the learned mask is as good as its density alone. Positions add nothing at the 1-px scale beyond what spacing, as imposed here, might add.
- **Top-k is not worse than a random draw** at this density (+0.035 dB vs the white mean, within noise), so determinism by itself costs nothing. The expected deficit did not appear.
- Hypothesis, untested: the brain-only gain arises because brain k-space samples are more compressible than knee at this density, or because brain trains on a harder mask structure; no experiment here separates these.

## Caveats

- **Legacy metric mask.** Not comparable with apodized-mask reports. A rescoring of the nine `best_model.pt` / `last_model.pt` files under the apodized mask has not been done.
- **Single training seed for the spacing arms** (42), two mask seeds, one top-k mask. The PD vs white difference rests on two paired comparisons; the 0.06 dB mean in seed 1 is within the noise floor.
- **Spacing acts only at 1 px.** The learned density is structured at pixel scale (outside radius 96 px the median probability is 0.004 but the mean is 0.056), so Poisson-disc spacing can act only between neighbours. A 4-px-blurred density family was tried and dropped because it formed 2-px lattices and could not be density-matched. The result says nothing about spacing at larger scales or about position-aware learners.
- **Ring-density mismatch.** PD and white masks match the learned density to within 6% per ring, except the outermost ring of `white_loupe_s1` (12.5%). Part of the PD-white gap could come from this residual density difference rather than spacing; not tested.
- **Final-epoch numbers**, not best-epoch. `vdinit_s1` has the transient dip at epoch 30 but is at 34.8+ by epoch 31.
- **Environment drift.** torch 2.12 / Python 3.14 and different GPUs vs the local reference (torch 2.6, A6000); the reference run agrees with the random-init arms to within 0.05 dB, so the drift appears small.
- **Uncommitted code.** The snapshot includes uncommitted changes to `trainer.py`, `sampler.py` and others, and the new samplers were not yet in a commit; `trained_at_commit` is therefore not recoverable from the runs. No `manifest.json` exists for this folder.
- **VD-init at one power (4)** and one clamp range; the paper's other settings were not tried.

## Figures

- `figures/masks_r8.png`: the five fixed masks and their density / spacing statistics (from `masks_r8.json`).

## Next steps

1. Rescore the nine checkpoints under the apodized metric mask and move the runs into `checkpoints/` so `report_manifest.py write` can index them.
2. Add seeds to the fixed-mask arms (more VarNet seeds and 3-4 more mask seeds per family) to settle the knee, where the PD-white gap is 0.00-0.09 dB against a 0.03 dB noise floor.
3. Remove the ring-density confound for the PD/white pair (tighter density matching, or a PD mask rejection-sampled to the white mask's ring counts).
4. If the brain gain holds, prototype a position-aware learner (BASS-style swaps) on brain first; if not, stop spending effort on spacing. Drop VD-init for LOUPE, or try a lower power or a short warm-up before releasing the prior.

## Reproduce

`bash reports/spacing_vdinit_r8/run.sh` regenerates `summary.md` and `summary.json` from the nine training logs (CPU, seconds). It reads `~/scratch/code/loupe_spacing_vdinit_r8/runs/`; override with `EXP=<path>`. To retrain, `cd ~/scratch/code/loupe_spacing_vdinit_r8 && sbatch --array=0-8%4 run_spacing_vdinit.sbatch` (about 1.5-3.5 h per task on one GPU; design notes in `run.md` there). `python scripts/report_manifest.py check reports/spacing_vdinit_r8` is expected to fail until step 1 under Next steps is done.
