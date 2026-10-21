# Are the CG-SENSE iteration count and Tikhonov weight tuned in the R=8 LOUPE runs? (2026-10-19)

> **Not yet in the report-folder convention** (no `manifest.json`, `run.sh` or provenance block; see `reports/README.md`). The metric mask is not recorded here: runs logged before 2026-10-20 used the legacy mask (`target > 0.1 * max`), and scripts migrated on 2026-10-20 use the apodized per-anatomy mask. Numbers are not comparable across the two, nor with `reports/metric_mask_reeval_r8/report.md`. Check the cited script and result timestamps before comparing, and rewrite with the `experiment-report` skill before citing this report.

## Question
The CG-SENSE LOUPE runs use 10 conjugate-gradient iterations and Tikhonov weight lamda = 0. No sweep of either was
recorded. With a trained mask and sensitivity net held fixed, how sensitive is validation quality to these two
settings, and is 10 / 0 near the best?

Conventions: lamda enters as (A^H A + lamda I) x = A^H y with k-space scaled so the RSS image peaks near 1;
PSNR/SSIM/NMSE are the trainer's signal-masked metrics, averaged per slice (so means differ from the trainer's
dataset-level PSNR).

## Runs compared
One trained model re-scored under different reconstructor settings (no retraining).

| Group | Path | Settings | Epochs / seed | Baseline val PSNR / SSIM (per-slice mean) |
|---|---|---|---|---|
| CG-SENSE, conv sens-net | `checkpoints/loupe2d_cgsense_r8_brain_knee_shared` | 10 iters, lamda 0 (training setting) | 50 / seed 42, R=8, LOUPE slope 10 | all 36.458 dB / 0.9082; brain 35.580 / 0.8994; knee 37.336 / 0.9170 |

## Methods
1. Load the checkpoint; for each setting set `model.reconstructor.num_iters` and `.lamda`, then score all 688
   validation slices (344 brain, 344 knee) with the model's own mask probabilities. Two Bernoulli mask draws per
   slice, identical for every setting (common random numbers). Script:
   `scripts/diagnostics/cgsense_hyperparam_sweep_r8.py`; output `reports/cgsense_hyperparams_r8/loupe2d_cgsense_r8_brain_knee_shared.json`.
2. Grid: iterations 3, 5, 10, 20, 40, 80 at lamda 0; lamda 1e-4, 1e-3, 1e-2, 1e-1 at 10 and 40 iterations.

## Results
PSNR in dB (delta against the 10 / 0 setting), SSIM in brackets:

| iters | lamda | all | brain | knee |
|---|---|---|---|---|
| 3 | 0 | 35.35 (-1.11) [0.9113] | 33.53 (-2.05) [0.8935] | 37.18 (-0.16) [0.9291] |
| 5 | 0 | **36.74 (+0.29)** [0.9158] | 35.34 (-0.24) [0.9022] | **38.15 (+0.81)** [0.9294] |
| 10 | 0 (training) | 36.46 [0.9082] | 35.58 [0.8994] | 37.34 [0.9170] |
| 20 | 0 | 34.30 (-2.16) [0.8829] | 33.40 (-2.18) | 35.20 (-2.14) |
| 40 | 0 | 30.86 (-5.60) [0.8381] | 30.01 (-5.57) | 31.72 (-5.62) |
| 80 | 0 | 26.04 (-10.42) [0.7681] | 25.64 (-9.94) | 26.43 (-10.90) |
| 10 | 1e-4 | 36.46 (+0.005) | 35.59 (+0.004) | 37.34 (+0.006) |
| 10 | 1e-3 | 36.50 (+0.044) [0.9089] | 35.62 (+0.037) | 37.39 (+0.051) |
| 10 | 1e-2 | 36.48 (+0.024) [0.9133] | 35.49 (-0.089) [0.9027] | 37.47 (+0.138) [0.9239] |
| 10 | 1e-1 | 30.81 (-5.65) [0.9065] | 29.52 (-6.06) | 32.10 (-5.24) |
| 40 | 1e-3 | 33.37 (-3.09) | 32.38 | 34.36 |
| 40 | 1e-2 | 36.20 (-0.26) [0.9096] | 35.23 (-0.35) | 37.16 (-0.18) |
| 40 | 1e-1 | 30.81 (-5.65) | 29.52 | 32.10 |

The 10-iteration and 40-iteration results at lamda 0.1 are identical, i.e. the solver has converged to the
regularised solution; that solution is 5.6 dB worse than the baseline.

Reading: without regularisation PSNR peaks at a few iterations and then falls steeply (80 iterations: -10.4 dB),
so the iteration count is acting as early-stopping regularisation. Ten iterations is close to the peak overall but
not at it: 5 iterations is better on knee (+0.81 dB) and worse on brain (-0.24 dB). A small lamda (1e-3 to 1e-2) at
10 iterations gives +0.02 to +0.05 dB overall and +0.003 to +0.007 SSIM at 1e-2, and with lamda 1e-2 the result is
stable out to 40 iterations (-0.26 dB) instead of collapsing.

## Conclusions
- 10 iterations / lamda 0 is not an arbitrary bad choice but also not tuned: the response to the iteration count is
  steep (several dB within a factor of 2), and brain and knee prefer different counts (brain about 10, knee about 5).
- A proper Tikhonov weight removes the sensitivity to the iteration count (lamda 1e-2: 10 iterations +0.02 dB, 40
  iterations -0.26 dB), at about equal PSNR to the baseline and higher SSIM.
- Hypothesis, untested: the learned mask and sensitivity net co-adapted to the truncated 10-iteration solver, so
  retraining under a regularised, converged solver (e.g. lamda 1e-2, 20 iterations) may give larger gains than this
  evaluation-only sweep shows.

## Caveats
- Evaluation only: mask and sensitivity net were trained at 10 / 0. A setting that scores equal or lower here is not
  excluded as a better training setting.
- One checkpoint (conv sens-net, seed 42); the bilinear run was not swept. Two mask draws per slice; no confidence
  intervals computed. The grid is coarse (no values between 5 and 10 iterations, nothing between lamda 1e-2 and 1e-1).
- Picking the best cell on the validation set that is also used for reporting is optimistic; deltas of 0.02-0.05 dB
  are within what I would expect from draw noise (about 0.05 dB for a per-slice delta, my estimate from the sens-net
  comparison) even though the common draws reduce it.
- The data here are noisy; the optimal lamda depends on the k-space scaling used by the loader.

## Figures
No figures. Numbers: `reports/cgsense_hyperparams_r8/loupe2d_cgsense_r8_brain_knee_shared.json`.

## Next steps
1. Finer grid: iterations 5-10, lamda 1e-3 to 3e-2, separately by anatomy, with a held-out half for selection.
2. Retrain one arm (seed 42, 50 epochs) at lamda 1e-2 and 20 iterations, compare against the 10 / 0 baseline and the
   second-seed baselines.
3. Repeat the sweep on the bilinear checkpoint to see whether the choice interacts with the upsampling.
