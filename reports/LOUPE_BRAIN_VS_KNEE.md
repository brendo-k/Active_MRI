# Learned LOUPE 2D masks: brain vs knee

## Setup
- Separate zero-step LOUPE 2D samplers (`training.num_step=0`, `sampler_type=loupe_2d`) for brain and knee, VarNet (12 cascades, learned sensitivity net) as recon, R=4 (25% budget).
- Full fastMRI training sets (T2 contrast for evaluation), 30 epochs, lr 1e-3 (step at 25), 3 seeds per anatomy, launched by `scripts/train_loupe_anatomy.sh` through the repo's `train_sequential.py`.
- Checkpoints are the best-validation-loss epoch (29 or 30 of 30), so the masks are essentially converged.
- Analysis: `scripts/analyze_loupe_anatomy.py` (patterns), `scripts/eval_flipped_mask.py` (orientation dependence). Outputs in `runs/loupe_anatomy/analysis/`.
- Mask = sigmoid(10 · logits), before the sampler's rescaling to the budget. Evaluation masks are random binary draws from it.

## 1. Are the learned patterns stable across seeds?
Mostly yes at large scale. Pearson r between seed pairs (mean [min, max]):

| Comparison | Brain (3 pairs) | Knee (3 pairs) | Brain vs knee (9 pairs) |
|---|---|---|---|
| Raw | 0.516 [0.514, 0.520] | 0.330 [0.318, 0.340] | 0.431 [0.321, 0.613] |
| Smoothed (σ=2) | 0.980 [0.974, 0.990] | 0.957 [0.936, 0.983] | 0.900 [0.893, 0.914] |
| Radial residual (smoothed) | 0.961 [0.949, 0.980] | 0.873 [0.813, 0.943] | 0.802 [0.785, 0.821] |

- Raw r is low because each seed starts from a different random initialisation, which leaves pixel-level speckle. Smoothing removes it.
- After smoothing, seeds agree strongly (r 0.94–0.99), so the broad layout is reproducible.
- Brain is more stable than knee, most clearly on the radial residual (0.95–0.98 vs 0.81–0.94). Knee's finer structure varies more between seeds.

## 2. Do the patterns differ between brain and knee?
Yes in character, but the pair-level separation is thin.

- Both masks are dense at the k-space centre and fall off with radius. Mean probability is 0.257 (brain) and 0.251 (knee).
- Knee has a sharper low-frequency peak in the radial profile.
- Brain is clearly asymmetric in kx (its right side is near zero). Knee is only mildly asymmetric. The ky profile is symmetric for both.
- Separation is marginal: on smoothed masks every seed pair beats every brain-knee pair, but only by 0.022. On the radial residual the distributions overlap (−0.008). Seed-to-seed variation in knee is of the same order as the brain-knee difference.

Conclusion: the anatomies differ mainly in the kx asymmetry and the radial profile, but the difference is not large compared with seed variability.

## 3. Does the reconstruction depend on the asymmetry?
Each trained model was re-evaluated on its own validation set with the mask logits edited. The reconstructor was kept fixed. ΔPSNR in dB vs the unmodified mask, mean over 3 seeds [min, max]:

| Condition | Brain | Knee |
|---|---|---|
| `orig_reseed` (noise floor) | 0.000 | 0.000 |
| `kx_flip` | −0.218 [−0.228, −0.199] | −0.064 [−0.101, −0.043] |
| `ky_flip` (control) | −0.002 | −0.006 |
| `point_flip` | −0.218 | −0.072 |
| `kx_symmetric` | −0.084 | −0.049 |

Baseline PSNR is about 29.9 dB (brain) and 31.2 dB (knee). Baseline SSIM is about 0.468 (brain) and 0.536 (knee).

- Flipping along kx hurts every seed, and by far more than the noise floor. Flipping along ky is harmless.
- The brain model depends on its orientation about 3.5× more than the knee model, consistent with its stronger asymmetry.
- Removing the asymmetry while keeping the budget (`kx_symmetric`) also hurts. The asymmetry is therefore useful and the mask and reconstructor have co-adapted to it.

## Caveats
- The effect sizes are small (≤0.22 dB). The evaluation shows the dependence exists but not why the brain asymmetry arises. Plausible causes are coil layout or phase-encode conventions in the data, or co-adaptation with VarNet. These were not tested.
- Three seeds per anatomy gives only 3 within-anatomy and 9 cross-anatomy pairs, so the stability and difference statistics are descriptive, not significance tests.
- Brain and knee validation sets differ in size and content, so absolute PSNR/SSIM should not be compared across anatomies.
- The `kx_flip` evaluation changes only the mask. A model retrained on the flipped layout might recover the loss.

## Figures
`runs/loupe_anatomy/analysis/`: `masks.png`, `mean_and_difference.png`, `profiles.png`, `correlation.png`. Numbers: `summary.json`, `flip_eval_{brain,knee}.json`.
