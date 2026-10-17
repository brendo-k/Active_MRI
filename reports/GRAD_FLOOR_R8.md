# Does flooring the LOUPE sigmoid/STE gradient change performance or the learned mask? (2026-10-14)

## Question
LOUPE-2D logits saturate during training: once |slope * w| is large, the sigmoid derivative is near zero and
that location stops receiving gradient. A *gradient floor* replaces the backward derivative
s(1 - s) with max(s(1 - s), floor * 0.25), i.e. at least `floor` times the sigmoid's peak derivative, for both
the mask sigmoid (`sampler.mask_grad_floor`) and the straight-through binarization surrogate
(`sampler.ste_grad_floor`). The forward pass is unchanged. This is a lower bound on the local derivative, not
gradient-norm clipping; the global norm clip (`GRAD_CLIP_NORM = 1.0`) is on in both runs and never fired.
Does a floor of 0.05 change reconstruction quality or the final sampling mask at R=8?

Conventions: k-space is DC-centred, 256 x 256, sampled point-wise (2D, not line-constrained). Mask
p(k) = sigmoid(slope * w) is the learned parameter before budget rescaling. The *effective* probability is
p(k) after the sampler's `RescaleProbMap` to mean 1/8, with the 10 x 10 preselected centre fixed at 1.
"Saturated" means |slope * w| > log 99, i.e. p outside [0.01, 0.99]. Plots label the axes kx / ky.

## Runs compared
Both runs: `train_sequential`, `sampler_type=loupe_2d`, `mask_train_mode=straight_through`, slope 5,
VarNet (12 cascades, `dc_mode=where`), `datasets=brain_knee_balanced` (1350 train slices per anatomy),
R=8, batch 16, lr 1e-3 (step to 1e-4 at epoch 40, 2 warm-up epochs), 50 epochs, `training.seed=42`.
Same seed and data order; the floor settings are the only config difference.

| Group | wandb | Checkpoint dir | Floors (mask / STE) | Best-val-loss epoch | Val brain PSNR / SSIM (epoch 49) | Val knee PSNR / SSIM (epoch 49) |
|---|---|---|---|---|---|---|
| floor 0.05 | `uo8d68n0` | `checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05` | 0.05 / 0.05 | 48 | 36.55 dB / 0.9328 | 34.26 dB / 0.9529 |
| no floor | `ijhnsmnm` | `checkpoints/grad_floor_compare/loupe2d_r8_slope5_nofloor` | 0 / 0 | 46 | 36.52 dB / 0.9325 | 34.25 dB / 0.9529 |

Val metrics are the trainer's background-masked aggregate metrics on the balanced val set (brain 23 volumes /
344 slices, knee 10 volumes / 344 slices), mean of 2 mask-draw streams (`reports/grad_floor_r8/eval.log`).
SSIM ~0.93-0.95, so both models are healthy. Wall-clock runtime 4.8 h each (`dynamics.json`).

## Methods
1. **Paired val evaluation** (`scripts/eval/compare_grad_floor_r8.py`, reuses `evaluate`/`paired_stats` from
   `scripts/eval/compare_anatcond_r8.py`): best-val-loss and last (epoch 49) checkpoints of each run, scored
   with the trainer's body-masked PSNR/SSIM on the full val set. Both models see the same slices and the same
   RNG stream for the Bernoulli mask draws; results are averaged over 2 draw streams (seeds 1234, 4321).
   Paired delta = per-slice metric(floor) minus metric(no floor); 95% CI = mean +/- 1.96 x std of
   per-volume mean deltas / sqrt(n volumes). Output: `reports/grad_floor_r8/paired.json`, `eval.log`.
2. **Ending masks** (`scripts/plots/plot_grad_floor_masks.py --checkpoint last_model.pt`, and `best_model.pt`):
   p(k), effective probability, floor minus no-floor difference, radial profile, logit histogram, saturation and
   "binariness" fractions. Overlap = share of the top-N effective-probability locations shared by both masks,
   N = expected sample count (8192). Output: `reports/grad_floor_r8/masks_{last,best}.{png,json}`.
3. **Training dynamics** (`scripts/plots/plot_grad_floor_dynamics.py`): unsampled wandb history (`scan_history`)
   of per-epoch val metrics, logit-saturation stats, and per-step sampler gradient norm and clip-firing rate.
   Output: `reports/grad_floor_r8/dynamics.{png,json}`, `dynamics_per_epoch.csv`.

## Results

### Reconstruction quality
| Comparison | Brain PSNR (dB), 95% CI | Brain SSIM | Knee PSNR (dB), 95% CI | Knee SSIM |
|---|---|---|---|---|
| floor - no floor, best ckpts (ep 48 vs 46) | +0.020 [+0.011, +0.029], 91% of vols | +0.0004 [+0.0000, +0.0007] | +0.022 [+0.002, +0.042], 80% of vols | -0.0000 [-0.0001, +0.0000] |
| floor - no floor, last ckpts (ep 49) | +0.017 [+0.006, +0.027], 83% of vols | +0.0003 [-0.0000, +0.0007] | +0.027 [-0.026, +0.079], 50% of vols | -0.0001 [-0.0001, -0.0000] |
| *reference: last - best, no floor* | +0.007 [+0.000, +0.014] | +0.0001 | +0.005 [-0.032, +0.043] | +0.0001 |

The floor run is ahead by about 0.02 dB PSNR on both anatomies, consistently across brain volumes; SSIM
differences are at most 0.0004 and of mixed sign (knee SSIM is 0.0001 lower at the last checkpoint). The
paired deltas use a per-slice data range and average over slices, so they differ from the gap between the
aggregate numbers in the runs table (dataset-level MSE, +0.03 dB brain). That is 2-3x the
best-to-last checkpoint jitter within a run, but an order of magnitude below the ~0.1-0.26 dB that needed
seed-swap controls to establish for anatomy-specific masks. The learning curves overlap for the whole run
(`dynamics.png`, top row); the last-5-epoch mean val PSNR differs by +0.03 dB (brain) and +0.007 dB (knee).

### Ending masks (epoch 49)
| Statistic | floor 0.05 | no floor |
|---|---|---|
| Effective p < 0.01 (dead locations) | 41.9% | 42.0% |
| Saturated logits (\|slope * w\| > log 99) | 40.5% | 39.9% |
| p > 0.99 (outside preselected centre) | 0.05% | 0.05% |
| Max \|slope * w\| | 22.5 | 8.3 |
| 99th percentile \|slope * w\| | 16.0 | 7.4 |
| Mean binary entropy of effective p (bits/location) | 0.308 | 0.313 |

| Similarity between the two ending masks | Value |
|---|---|
| Pearson correlation of effective p | 0.991 |
| Sum \|diff\| / sum of both | 5.3% |
| Top-8192 location overlap | 93.7% |

The two masks are nearly the same: the same centre-dense variable-density shape with the same radial profile
(`masks_last.png`, top right) and the same ~42% of k-space effectively never sampled (the outer
diamond-shaped region visible in the p(k) maps, including the corners beyond |k| ~ 1.1 k_max). The visible differences are scattered single-location changes, mostly at mid frequencies. The
best-checkpoint masks give the same numbers to within 0.2 percentage points (`masks_best.json`).

### What the floor did to training
| Statistic | floor 0.05 | no floor |
|---|---|---|
| Saturated logit fraction, epoch 17 / 49 | 15.8% / 40.5% | 14.0% / 39.9% |
| Max \|slope * w\|, epoch 17 / 49 | 12.7 / 22.5 | 7.3 / 8.3 |
| Saturated logits that moved further out over the last epoch | 93% | 94% |
| Global norm clip firing rate | 0% | 0% |
| Median sampler grad norm, last epoch | 8.5e-5 | 8.2e-5 |

The floor works as built: saturated logits keep receiving gradient, which shows as max |slope * w| growing
about linearly (0.43 per epoch) until the lr step at epoch 40, while the no-floor run plateaus near 8. The
floor rescales the backward derivative but never flips its sign, so the outward-moving fraction is the same
by construction and is not evidence on its own. The informative number is that, despite the extra gradient,
the saturated and dead fractions finish within 0.6 percentage points of the no-floor run. Almost all the extra
logit mass is on the negative side (logit histogram in `masks_last.png`: tail to -22 vs a hard edge near -8).

## Conclusions
- A 0.05 gradient floor gives at most a small gain at R=8: about +0.02 dB PSNR on brain and knee (CI excludes
  zero for brain at both checkpoints and for knee at the best checkpoint), with SSIM differences of 0.0004 or
  less and of mixed sign. This is far below what has counted as a meaningful mask effect in this project, and
  it comes from one seed.
- The ending sampling masks are essentially unchanged (correlation 0.991, 94% top-budget overlap, identical
  radial profile and dead-region size).
- Keeping gradient flowing to saturated logits did not change how many locations end up switched off
  (dead fraction 41.9% vs 42.0%). This is consistent with saturation here being mostly the loss saying
  "don't sample here" rather than a vanishing-gradient trap. Hypothesis, untested: no test tracked whether
  individual locations re-entered (0.01, 0.99) in either run.
- Hypothesis, untested: the no-floor plateau at max |slope * w| ~ 8 may be an Adam stall rather than a
  converged optimum. The mask logits share the default Adam group (lr 1e-3, eps 1e-8; only autosamp gets
  `POSITION_ADAM_EPS`), so once the sigmoid derivative makes their gradient fall below eps, their steps shrink.
  With the floor, the gradient stays above eps and Adam takes near-lr-sized steps, matching the linear growth.
- The floor's only clear side effect is unbounded growth of the negative logits (|slope * w| up to 22.5 and
  still rising before the lr drop). This is harmless for p(k) but worth noting for longer runs.

## Caveats
- Single training seed per arm. The paired CI covers val-set and mask-draw variation only, not training
  variation. With cuDNN non-determinism there is no same-config repeat to say how much of the 6% mask
  difference or the 0.02 dB is from the floor versus run-to-run noise.
- One floor value (0.05), one slope (5), one acceleration (R=8), one sampler (LOUPE-2D point-wise). Slope 5
  is lower than the slope 10 of earlier R=8 runs; a higher slope saturates faster and might give the floor
  more to do.
- Best checkpoints come from different epochs (48 vs 46); the last-checkpoint comparison is like-for-like and
  gives the same answer.
- The per-epoch mask stats are aligned to epochs by row order in the wandb history. Epoch-17 and epoch-49
  values match the corresponding checkpoints exactly.

## Figures
- `reports/grad_floor_r8/masks_last.png`: p(k), effective probability, difference map, radial profile, logit
  and probability histograms at epoch 49 (`masks_best.png`: same for best-val checkpoints).
- `reports/grad_floor_r8/dynamics.png`: val PSNR/SSIM per anatomy, saturation fraction, max |logit|,
  fraction of saturated logits moving outward, smoothed sampler gradient norm (values in
  `dynamics_per_epoch.csv`).

## Next steps
1. Repeat both arms with a second training seed (or a same-config repeat) to set a noise floor before
   reading anything into the 0.02 dB.
2. If the floor is to be kept, test it where saturation should hurt: slope 10-20, or a schedule where the
   optimal mask moves late (e.g. a curriculum over R), and track how many saturated logits re-enter (0.01, 0.99).
3. Test the Adam-stall hypothesis: log the saturated logits' gradient magnitude against eps, or rerun the
   no-floor arm with eps 1e-16 on the mask group and see whether its max |slope * w| tracks the floor run.
4. Compare against capping the logits (e.g. clamp |slope * w| <= log 99 after each step), which bounds the
   negative drift and keeps every location one gradient step away from re-entering.
