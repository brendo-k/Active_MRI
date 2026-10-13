# Vertical streaks in learned brain LOUPE masks: where do they come from? (2026-10-05)

## Question
The learned brain LOUPE 2D mask has narrow vertical streaks: specific columns, spaced about 10-20 apart, carry
consistently more probability. Are they noise, a property of the k-space data, or something the loss asks for?
Is it a phase effect (they disappear when the image phase is scrambled)?

Axis convention: mask columns are the **phase-encode** axis. fastMRI brain k-space is 768 (readout, rows) x 396
(phase-encode, columns); the loader decimates the readout (rows) by 2, then centre-crops both axes to 256.
The plots label columns "kx".

## Runs compared (all `train_sequential.py`, LOUPE 2D, zero-step, R=4, 30 epochs, lr 1e-3, VarNet)
| Group | Path | Cascades | Seeds |
|---|---|---|---|
| brain, 12 casc | `runs/loupe_anatomy/brain` | 12 | 3 |
| knee, 12 casc | `runs/loupe_anatomy/knee` | 12 | 3 |
| brain, 1 casc | `runs/loupe_casc1/brain` | 1 | 3 |
| 1 casc, no sense mask | `runs/loupe_casc1_nosensemask/brain` | 1 | 1 |
| 1 casc, phase scramble 1.0 / 1.5 rad | `runs/loupe_casc1_phasescramble{,_strong}/brain` | 1 | 3 each |

Validation (trainer's background-masked metrics, full val set, mean over 3 seeds): brain 34.72 dB / SSIM 0.944
(587 slices), knee 31.54 dB / SSIM 0.936 (338 slices). Figures from `scripts/plot_loupe_anatomy_recon.py` and
`scripts/plot_mean_masks.py` in `runs/loupe_anatomy/analysis/`.

## Methods
All outputs go to `runs/loupe_anatomy/analysis/`. Mask p(k) = sigmoid(10 * logits), before budget rescaling.
"Ripple" = mean of p over rows, minus its 13-column moving average. z-score = value minus mean, over standard
deviation, within one curve (used only to overlay curves with different units).

1. **Reproducibility and specificity** (`scripts/plot_column_ripple.py`): ripple per seed for every group;
   Pearson r between seeds and the dominant period from an FFT over columns 8-120.
2. **Does it matter?** Rescale each column of the trained brain mask to remove its ripple (or shuffle the ripple
   onto other columns), reconstructor fixed, re-evaluate on the val set with the trainer's metrics.
3. **Is it in the data?** (`scripts/check_kspace_column_ripple.py`) Over the 587 brain val slices (38 scans), per
   column: mean log coil-RSS |k|, and column-to-column phase coherence
   |sum k_x conj(k_{x-1})| / sum |k_x||k_{x-1}|. Compared with the mask ripple for the dataset mean and per scan.
4. **Phase-coherence comb** (`scripts/coherence_comb_test.py`): dips in dataset-mean coherence vs mask peaks;
   fold coherence at period 19 and rank the boundary phase among the 19 phases, for several column windows.
5. **What does the loss want?** (`scripts/loupe_column_gradient.py`, `scripts/plot_column_gradient.py`) Seed 0
   brain model. Accumulate dLoss/dp(k) over the val set (straight-through gradient, divided by
   slope * p * (1 - p)), sum over rows. Probed at the trained mask and at the destreaked mask. Reproducibility =
   Pearson r between gradients from two halves of the val batches. Repeated with the val k-space phase-scrambled
   (1 rad, 6-16 cycles/FOV; out of distribution for this reconstructor).

## Results
**Streaks are systematic, brain-specific and phase-dependent** (detrended column profile, columns 8-120):

| Group | Ripple RMS | Seed-to-seed r | Dominant periods (columns) |
|---|---|---|---|
| brain, 12 casc | 0.027 | 0.93-0.94 | 9.3, 10.2, 18.7 |
| brain, 1 casc | 0.015 | 0.78-0.82 | 9.3 |
| 1 casc, no sense mask | 0.015 | n/a (1 seed) | 9.3 |
| 1 casc, phase scramble 1.0 | 0.009 | 0.31-0.58 | none (noise) |
| 1 casc, phase scramble 1.5 | 0.007 | about 0 | none (noise) |
| knee, 12 casc | 0.006 | 0.51-0.59 | none (noise) |

Related layout facts (mean mask, 3 seeds): both anatomies are dense on the left of k-space and sparse on the
right; the brain is more one-sided (share of mask mass on the right half 0.28 vs knee 0.38). The 12-cascade brain
mask has no big fully-sampled centre; the 1-cascade mask does.

**They are harmless:** flattening the ripple changes val PSNR by -0.003 to +0.002 dB (3 seeds, no reseed noise
floor measured). Moving the same ripple to other columns costs 0.002-0.046 dB depending on seed.

**Not visible in the mean k-space:** detrended mean log |k| has ripple RMS 0.003 and r = +0.10 with the mask
ripple (+0.15 for the 1-cascade masks); per-scan median r = +0.02, 3% of scans have |r| > 0.3. A whole-profile
correlation of coherence with the ripple is -0.00.

**But coherence dips line up with half of the mask peaks, in one window:** mean coherence dips at columns
24, 43, 62, 81, 100 (period 19); mask peaks sit one column earlier (23, 42, 61, 80, 99), as expected for a
break between columns x-1 and x. The other peaks (14, 33, 52, 72, 91, 109) fall midway between.

| Window | Boundary-phase coherence | Rank among 19 phases |
|---|---|---|
| columns 20-105 | -0.0131 (others +0.0005 +/- 0.0019) | 1 |
| columns 8-120 | -0.0045 | 3 |
| columns 5-250 | -0.0026 | 7 |
| columns 130-250 | -0.0020 | 9 |

Per scan in 20-105, 82% of the 38 scans have below-average coherence at the boundary columns (median z -0.66).

**The loss itself asks for the streaks, and only with intact phase** (brain seed 0):

| Mask state / val data | Split-half r of gradient | r(gradient, mask ripple) |
|---|---|---|
| trained | 0.27 | +0.18 |
| destreaked | 0.88 | **-0.73** |
| destreaked, phase-scrambled val data | 0.92 | +0.22 |

Negative = raising probability at the streak columns lowers the loss. With scrambled phase the gradient is flat
over columns 8-100; the remaining correlation comes from large excursions near the centre (columns above 100).

## Conclusions
- The streaks are a reproducible feature of the brain masks: shared across seeds, cascade count and sense mask,
  mostly absent in the knee, and removed by phase scrambling.
- They are not a simple amplitude pattern in the k-space. Evidence for a phase origin: the loss gradient favours
  the streak columns on the original data but not on phase-scrambled data, and the 19-column coherence dips line
  up with half of the peaks.
- Practical effect is negligible (about 0.003 dB), so they need no fixing.
- Hypothesis, untested: the brain T2 scans are turbo spin-echo (header: `TurboSpinEcho`, TE 113 ms, echo spacing
  10.26 ms, i.e. 11 echoes). An echo train of 21 would put echo 11 at the centre and give phase-encode blocks
  about 19 lines wide, with boundaries at 128 +/- 9.5 + 19k, which matches the dips. The header does not state
  the echo-train length.

## Caveats
- The coherence comb was chosen after looking at the dataset-mean plot, so the column window is post hoc. The
  full-range test (columns 5-250) is not significant, and the effect is absent on the right of k-space.
- The gradient test is one seed and probes a reconstructor co-adapted to the mask. The phase-scrambled probe is
  out of distribution for it. The gradient is smoother than the mask ripple and reproduces its broad bumps, not
  every spike.
- Phase was only measured as column-to-column coherence; slow phase trends and row-wise relationships were not.
- The knee phase coherence was not computed, and its phase-encode axis was assumed to match the brain's.
- Brain-vs-knee absolute metrics are not comparable (different val sets).

## Figures (`runs/loupe_anatomy/analysis/`)
`recon_{brain,knee}_seed0.png` (mask, val slices, recon, error), `mean_masks_p_k.png` (mean p(k) and p(k) - p(-k)),
`column_ripple.png` (per-seed ripple), `kspace_column_ripple.png` (data mean, coherence, per-scan r),
`column_gradient.png` (loss gradient vs ripple, with and without phase scramble).

## Next steps
- Repeat the gradient test on seeds 1-2 and on the 1-cascade runs.
- Compare coherence per phase-encode line against the TSE echo index if the echo-train length can be obtained
  from the scanner protocol.
- Train a 1-cascade knee LOUPE (none exists) for a like-for-like brain/knee comparison.
