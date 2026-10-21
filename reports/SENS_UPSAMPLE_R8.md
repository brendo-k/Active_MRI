# Does bilinear upsampling in the sensitivity U-Net remove the early edge artefact in LOUPE masks? (2026-10-19)

> **Not yet in the report-folder convention** (no `manifest.json`, `run.sh` or provenance block; see `reports/README.md`). The metric mask is not recorded here: runs logged before 2026-10-20 used the legacy mask (`target > 0.1 * max`), and scripts migrated on 2026-10-20 use the apodized per-anatomy mask. Numbers are not comparable across the two, nor with `reports/metric_mask_reeval_r8/report.md`. Check the cited script and result timestamps before comparing, and rewrite with the `experiment-report` skill before citing this report.

## Question
During LOUPE training the probabilistic mask shows bright blobs at the k-space borders (top/bottom and left/right
edge midpoints). The suspected cause is the transposed-convolution upsampling in the coil-sensitivity U-Net.
`sampler.sens_upsample_method=bilinear` replaces it with bilinear interpolation. Does that remove the blobs, and does
it change reconstruction quality? The comparison is matched 50-epoch CG-SENSE runs, two seeds per arm (42 and 43).

Conventions: masks are 256x256 k-space, centre = DC; "probabilistic mask" is the sampler's p(k) before binarisation
(bottom-left panel of the trainer's training visualisation, one training batch per epoch).

## Runs compared
Both: `train.py --model loupe --recon cg_sense --dataset brain_knee_balanced --acceleration 8 --batch-size 16
--epochs 50 --seed {42,43}`. CG-SENSE 10 iterations, lambda 0; LOUPE 2D, slope 10, grad floors 0, lr 1e-3 with step
schedule, 2 seeds per arm. Brain + knee, balanced. The seed-43 runs were launched by `analysis/sens_upsample/run_seed43.sh`.

| Group | Path | wandb | `sens_upsample_method` | Final brain PSNR / SSIM | Final knee PSNR / SSIM |
|---|---|---|---|---|---|
| conv (default) | `checkpoints/loupe2d_cgsense_r8_brain_knee_shared` | xb6tna6g | conv | 34.92 dB / 0.9131 | 32.70 dB / 0.9291 |
| bilinear | `checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep` | wcxaz648 | bilinear | 34.93 dB / 0.9130 | 32.71 dB / 0.9290 |
| conv, seed 43 | `checkpoints/loupe2d_cgsense_r8_brain_knee_shared_seed43` | - | conv | 34.92 dB / 0.9131 | 32.71 dB / 0.9291 |
| bilinear, seed 43 | `checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep_seed43` | - | bilinear | 34.92 dB / 0.9129 | 32.72 dB / 0.9292 |

Rows without a seed label are seed 42. The resolved configs in the seed-42 log headers differ only in `sens_upsample_method` and `out_dir`, plus three options
added to the code after the conv run (`sense_cache_dir=null`, `selection=bernoulli`, `dcm_*`); all are at
their defaults, which I expect to leave training unchanged (not verified by re-running the old code).
Validation numbers are the trainer's own (background-masked) metrics, parsed from the logs.

## Methods
1. Validation metrics per epoch: parsed from the training logs (the metric line precedes each `Epoch = [n/50]`
   line); n = 344 validation slices per anatomy. Script: `scripts/diagnostics/sens_upsample_compare_r8.py` (`--seed 43` for the second seed).
2. Mask comparison: crop the probabilistic-mask panel from `visualizations/train_epoch=N_iter=1_step0.png` at epochs
   1, 2, 3, 5, 10, 20, 50 for both runs and tile them (same script). This is a visual comparison of one training
   batch per epoch; no border statistic was computed.

## Results
Validation PSNR (dB) / SSIM at selected epochs:

| Epoch | brain conv | brain bilinear | knee conv | knee bilinear |
|---|---|---|---|---|
| 1 | 32.05 / 0.8837 | 32.37 / 0.8876 | 30.63 / 0.8976 | 30.87 / 0.9021 |
| 2 | 33.54 / 0.8982 | 33.83 / 0.9013 | 31.43 / 0.9080 | 31.59 / 0.9106 |
| 5 | 34.31 / 0.9062 | 34.42 / 0.9074 | 31.97 / 0.9161 | 32.07 / 0.9175 |
| 10 | 34.58 / 0.9093 | 34.65 / 0.9099 | 32.30 / 0.9215 | 32.34 / 0.9221 |
| 20 | 34.78 / 0.9116 | 34.82 / 0.9118 | 32.54 / 0.9256 | 32.57 / 0.9264 |
| 50 | 34.92 / 0.9131 | 34.93 / 0.9130 | 32.70 / 0.9291 | 32.71 / 0.9290 |

Seed 43 (conv / bilinear):

| Epoch | brain conv | brain bilinear | knee conv | knee bilinear |
|---|---|---|---|---|
| 1 | 32.01 / 0.8836 | 32.34 / 0.8874 | 30.62 / 0.8975 | 30.93 / 0.9027 |
| 2 | 33.59 / 0.8990 | 33.83 / 0.9013 | 31.53 / 0.9092 | 31.61 / 0.9107 |
| 5 | 34.36 / 0.9068 | 34.40 / 0.9072 | 32.07 / 0.9173 | 32.04 / 0.9173 |
| 10 | 34.58 / 0.9093 | 34.64 / 0.9100 | 32.32 / 0.9217 | 32.33 / 0.9219 |
| 20 | 34.80 / 0.9117 | 34.84 / 0.9121 | 32.55 / 0.9258 | 32.59 / 0.9266 |
| 50 | 34.92 / 0.9131 | 34.92 / 0.9129 | 32.71 / 0.9291 | 32.72 / 0.9292 |

Bilinear leads by 0.3 dB (brain) and 0.24-0.31 dB (knee) after epoch 1 in both seeds, and the lead shrinks to
+0.00-0.01 dB at epoch 50. Seed-to-seed differences within one arm at epoch 50 are 0.00-0.01 dB (conv: brain
34.92 vs 34.92, knee 32.70 vs 32.71), the same size as the conv-bilinear gap.

Masks (`reports/sens_upsample_r8/mask_montage.png` for seed 42, `mask_montage_seed43.png` for seed 43; top row conv,
bottom row bilinear):
- Conv: from epoch 2 there are bright blobs on a regular lattice at the borders (top/bottom and left/right edge
  midpoints, smaller ones along the edges). Clearest at epochs 2-5; the top/bottom blobs are still visible at epoch 10;
  gone by epoch 20.
- Bilinear: a smooth centre-weighted blob that grows into a diamond support; no border blobs at any inspected epoch.
- Both runs show a faint horizontal streak along the central line, and the epoch-50 masks look alike.
- Seed 43 reproduces this: conv has border blobs from epoch 2 (visible to epoch 10, gone by epoch 20), bilinear has none.

## Per-slice comparison of the final models, seed 42 (688 validation slices)
Scripts: `scripts/diagnostics/sens_upsample_slice_diff_r8.py` (scores every sense-net x mask pair on common seeded
Bernoulli draws, 4 draws per slice, trainer-path PSNR) and `scripts/diagnostics/sens_upsample_slice_diff_summary_r8.py`.
Outputs: `reports/sens_upsample_r8/slice_diff.json`, `slice_diff_summary.json`, `slice_diff_psnr.npz`, `final_masks.pt`.
Delta = bilinear minus conv, dB per slice, mean +/- sd over slices. Draw noise: sd of the difference between two
single-draw scorings of the same conv model is 0.098 dB, so about 0.05 dB for a difference of two 4-draw means
(my estimate, not measured directly).

| Subset | n | full (own mask + sense net) | mask only | sense net only |
|---|---|---|---|---|
| all | 688 | +0.016 +/- 0.188 | +0.029 +/- 0.047 | -0.013 +/- 0.184 |
| brain | 344 | +0.001 +/- 0.080 | +0.022 +/- 0.046 | -0.021 +/- 0.065 |
| knee | 344 | +0.031 +/- 0.252 | +0.036 +/- 0.047 | -0.006 +/- 0.251 |
| knee PD | 135 | -0.016 +/- 0.213 | +0.050 +/- 0.047 | -0.070 +/- 0.208 |
| knee PDFS | 209 | +0.062 +/- 0.271 | +0.027 +/- 0.045 | +0.035 +/- 0.268 |
| support, lowest third | 229 | +0.002 +/- 0.173 | +0.038 +/- 0.051 | -0.037 +/- 0.170 |
| support, highest third | 229 | +0.064 +/- 0.247 | +0.016 +/- 0.043 | +0.049 +/- 0.238 |

Support = fraction of pixels inside the trainer's signal mask (target > 0.1 of its max; mean 0.36 brain, 0.37 knee).
Corrected 2026-10-20: the first version of the two support rows (and the 0.12 below) came from a per-row instead of per-slice threshold
(`support_recomputed.json`). Correlation of the full delta with support 0.15,
with slice position 0.06, with conv PSNR 0.02.

Final masks (shared mask, probability before budget rescaling): pixelwise correlation 0.854; after a Gaussian
blur of 2 px, 0.998 (8 px, 0.9996); share of probability by radius within 1 percentage point of each other; mean
probability 0.151 conv vs 0.143 bilinear; mean absolute difference 0.053 (35% of the mean), max 0.99; mean
probability in the outer 24 px band 0.068 vs 0.066.

Spread across slices (`slice_diff_summary.json`):

| | brain | knee |
|---|---|---|
| slices with abs(delta) > 0.1 / 0.25 / 0.5 dB | 14% / 1.5% / 0% | 40% / 15% / 6% |
| slices where bilinear is better | 51% | 46% |
| share of delta variance between volumes | 25% | 29% |
| sd of per-volume means / within-volume sd | 0.04 / 0.07 | 0.13 / 0.21 |

The knee mean of +0.031 dB comes from one volume: `file1001856` (PDFS, 39 slices) has mean +0.41 dB
(sd 0.42, slices 18-24 at +0.85 to +1.40 dB). The other nine knee volumes average -0.017 dB (sd 0.17); their
volume means lie between -0.049 and +0.014. Largest single-slice losses for bilinear are -0.8 to -0.5 dB, mostly on the
first slices (1-3) of knee volumes at very high PSNR (45-52 dB).

Reading:
- The final masks differ in where individual k-space points sit, not in density: they agree once blurred by 2 px.
  Swapping only the mask changes slices by +0.03 +/- 0.05 dB, which is about the size of draw noise, with a small
  consistent positive mean (standard error 0.002 dB over 688 slices; bilinear mask slightly better).
- The per-slice differences that are larger than draw noise come from the sensitivity net (sd 0.18 dB, 0.25 dB in
  knee); the seed-variation control below shows that a second training of the same recipe differs by as much. They are of both signs and cancel in the mean: brain is nearly unaffected (sd 0.065 dB), knee slices with
  large support move by up to +/-1 dB.
- Why the mean stays equal: matched sampling density plus sense-net differences that are slice-specific and signed
  both ways. Whether the sense-net differences reflect better or worse coil-map estimates was not checked
  (no reference maps were compared); Hypothesis, untested.

## Other metrics and the sensitivity maps themselves (688 validation slices)
Scripts: `scripts/diagnostics/sens_upsample_sense_maps_r8.py` (per-slice metrics and map diagnostics) and
`scripts/diagnostics/sens_upsample_sense_maps_summary_r8.py` (drivers, extreme slices). Outputs:
`reports/sens_upsample_r8/sense_maps_per_slice.npz`, `sense_maps_summary.json`, `sense_maps_drivers.json`.
Per-slice means over 4 Bernoulli draws, trainer-path (signal-masked) metrics; the mean over slices differs from the
trainer's dataset-level PSNR (36.46 vs ~34.9 dB for conv) because of the different aggregation.

Definitions. Body mask: the sensitivity net's `pre_network` mask (RSS of the 10x10 centre of k-space above 10% of its
max); it has no weights. Fit: ||y - S x|| / ||y|| inside the body mask for the fully sampled coil images y, with
x = sum_c conj(S_c) y_c (0 = maps explain the data exactly). Oracle NMSE: NMSE of that fully sampled SENSE image
against the RSS target inside the signal mask. High-frequency share: fraction of the 2D-FFT energy of |S| beyond 12
cycles per field of view. No reference (ESPIRiT/BART) maps exist for these knee files, so all map diagnostics are
reference-free.

| Metric (mean over slices) | conv | bilinear | delta +/- sd of per-slice delta |
|---|---|---|---|
| PSNR dB (all) | 36.459 | 36.475 | +0.016 +/- 0.188 |
| SSIM (all) | 0.9082 | 0.9081 | -0.0001 +/- 0.0013 |
| NMSE (all) | 0.01731 | 0.01730 | -0.00002 +/- 0.00042 |
| fit residual, brain / knee | 0.150 / 0.212 | 0.145 / 0.209 | -0.0053 / -0.0024 |
| oracle NMSE, brain / knee | 0.01045 / 0.01134 | 0.01037 / 0.01127 | -0.00008 / -0.00006 |
| high-frequency share of abs(S), brain / knee | 0.0347 / 0.0461 | 0.0315 / 0.0435 | -0.0032 / -0.0027 |
| signal pixels outside the body mask | 3.8% | 3.8% | 0 (identical in every slice) |

Relative difference of abs(S) between the two nets inside the body mask: 0.14 (brain), 0.16 (knee).

Which slices move with what (correlation of the per-slice PSNR delta with the sense-map deltas):

| | brain | knee |
|---|---|---|
| fit residual delta | -0.34 | -0.60 |
| oracle NMSE delta | -0.36 | -0.64 |
| baseline PSNR of the conv run | +0.10 | -0.02 |

Knee slices in the top 20% of conv PSNR (n=69, mean 43.1 dB): delta -0.009 +/- 0.29 dB; bottom 20% (n=69, mean
30.2 dB): -0.004 +/- 0.26 dB. The 20 largest losses and gains are in `sense_maps_drivers.json`. On the slices where
bilinear gains (e.g. `file1001856` slices 17-24, +0.8 to +1.4 dB) its fit residual and oracle NMSE both fall; on
most of the large losses (e.g. `file1000662` slice 1, -0.81 dB, conv 52.1 dB) they rise, but by tiny absolute amounts
(oracle NMSE +0.0003 against a conv NMSE of order 1e-3). SSIM changes by less than 0.001 on all of these losses
except a few knee slices at -0.003.

Reading:
- The body masks are identical, so none of the difference comes from the sense mask; both runs zero the same 3.8% of
  signal pixels.
- The bilinear maps are not poorer on average: they fit the coil images slightly better (residual 2% lower, oracle
  NMSE 0.7% lower) and carry less high-frequency energy. The slices that gain are the ones where its maps fit better,
  and the slices that lose are the ones where they fit worse; both directions occur.
- Large losses on very high-PSNR slices are a scale effect of dB: the absolute error is tiny there, so a small absolute
  change in NMSE is a large change in dB; SSIM and absolute NMSE barely move. There is no systematic loss at high PSNR
  (top-quintile mean -0.009 dB). Hypothesis, untested: end slices of knee volumes are low-signal, so the maps there
  rest on a weak 10x10 centre and differ most between the two nets (relative map difference up to 0.29 in these slices).

## Second seed: is the per-slice difference more than seed-to-seed variation?
Script: `scripts/diagnostics/sens_upsample_seeds_r8.py`; outputs `reports/sens_upsample_r8/seeds_comparison.json`,
`seeds_comparison_psnr.npz`. All four models scored on the 688 validation slices with common Bernoulli draws (4 per
slice), each with its own mask and sensitivity net. Per-slice delta in dB, mean +/- sd over slices.

Mean per-slice PSNR (all / brain / knee): conv42 36.459 / 35.581 / 37.337; bilinear42 36.475 / 35.582 / 37.368; conv43
36.475 / 35.568 / 37.383; bilinear43 36.486 / 35.569 / 37.403.

| Contrast (all slices) | mean | sd | slices with abs > 0.25 dB |
|---|---|---|---|
| upsampling, seed 42 (bilinear - conv) | +0.016 | 0.188 | 8% |
| upsampling, seed 43 (bilinear - conv) | +0.011 | 0.177 | 9% |
| seed effect, conv (43 - 42) | +0.016 | 0.178 | 9% |
| seed effect, bilinear (43 - 42) | +0.011 | 0.127 | 6% |

By anatomy, sd of the upsampling contrast (seed 42 / 43) against the conv seed contrast: brain 0.080 / 0.119 vs 0.106;
knee 0.252 / 0.220 vs 0.224.

- The spread of per-slice differences between conv and bilinear is the same as between two seeds of the same arm
  (0.18 vs 0.18 dB overall), so most of the slice-level difference seen in one pair is ordinary run-to-run variation.
- The mean upsampling effect is +0.016 and +0.011 dB in the two seeds, the same size as the conv seed effect
  (+0.016 dB).
- Some of the slice-level effect repeats: the correlation of the upsampling contrast across seeds is 0.46 (brain 0.42,
  knee 0.47), against 0.25 for the conv and bilinear seed effects. Of the 31 slices with abs > 0.25 dB in both seeds, 84%
  have the same sign (knee 29 slices, 83%; brain 2 of 2); independent noise would give about 50%.
- `file1001856` (knee PDFS) is the volume that gains most in both seeds: +0.41 dB (seed 42) and +0.22 dB (seed 43).
  Brain `AXT2_202_2020016` loses in both (-0.117, -0.115 dB); the other brain volumes are within about +/-0.07 dB.

## Against the ESPIRiT (BART ecalib) masks (2026-10-20, seed 42, 688 validation slices)
Scripts: `scripts/diagnostics/sens_upsample_vs_ecalib_r8.py`, `sens_upsample_vs_ecalib_figure_r8.py`. Outputs:
`reports/sens_upsample_r8/ecalib_comparison.json`, `ecalib_comparison_psnr.npz`, `ecalib_comparison_masks.png`.
Runs: `loupe2d_cgsense_r8_brain_knee_ecalib_c10` / `_c24` (same recipe and seed as `conv`, but cached BART ecalib maps
instead of the sense U-Net; preselected centre 10x10 / 24x24). Each mask is drawn by its owner's sampler on common seeded
Bernoulli draws (4 per slice) and scored with (a) its own maps, (b) the cached ecalib maps, (c) the conv run's sense net.
The ecalib maps come from a 24x24 calibration, so the 10x10-centre masks (conv, bilinear, c10) leak unsampled k-space into them.

Final validation (trainer metric, brain / knee PSNR): conv 34.92 / 32.70, bilinear 34.93 / 32.71, ecalib_c10 35.28 / 33.30,
ecalib_c24 35.30 / 33.25.

Delta in dB per slice, mean +/- sd over slices (all / brain / knee):

| Contrast | all | brain | knee |
|---|---|---|---|
| own, bilinear - conv | +0.016 +/- 0.19 | +0.001 +/- 0.08 | +0.031 +/- 0.25 |
| own, ecalib_c10 - conv | +0.63 +/- 1.29 | +0.23 +/- 0.65 | +1.02 +/- 1.61 |
| ecalib maps on every mask: bilinear mask - conv mask | +0.020 +/- 0.05 | +0.017 | +0.023 |
| ecalib maps on every mask: c10 mask - conv mask | +0.265 +/- 0.27 | +0.298 | +0.232 |
| conv net on every mask: c10 mask - conv mask | +0.179 +/- 0.22 | +0.219 | +0.139 |
| maps only, conv mask: ecalib - conv net | +0.36 +/- 1.32 | -0.07 | +0.79 |
| own, ecalib_c24 - ecalib_c10 | -0.107 +/- 0.13 | -0.082 | -0.132 |

- Masks: conv, bilinear and ecalib_c10 pixelwise correlation 0.82-0.85 (c10 vs conv 0.84), 0.96-0.998 after a 2 px blur (c10 vs conv 0.971,
  c10 vs bilinear 0.964, conv vs bilinear 0.998); all four are dotted lattices of near-binary p(k). The ecalib_c10 mask puts more probability at
  radius 10-60 px and less in the outer ring and diamond edge than conv (outer-share 0.564 vs 0.610).
- The own-mask gap ecalib_c10 - conv is almost entirely the maps: per-slice correlation with the maps-only gap 0.98, with the mask-only gap -0.01.
  It concentrates in knee (+1.02 dB, 80% of slices positive, volumes up to +2.4 dB) and is absent in brain (maps-only -0.07 dB).
- With the maps held fixed the ecalib_c10 mask is still +0.27 dB (ecalib maps) / +0.18 dB (conv net) over the conv mask, in both anatomies,
  while the bilinear-vs-conv mask difference is +0.02-0.03 dB. So the bilinear mask matches the conv mask in quality; the ecalib mask is a
  genuinely different, better solution under both reconstructors.
- Per-slice bilinear-conv and ecalib_c10-conv gaps are uncorrelated (r = 0.03): the slice-level bilinear difference is not a precursor of the ecalib gain.

Caveats: one seed per run; the ecalib mask was co-trained with the ecalib maps, so its +0.27 dB under those maps includes co-adaptation (the
conv-net scoring, where the conv mask is the co-adapted one, still gives +0.18 dB); the calibration leakage above favours no mask
specifically (all three 10x10-centre masks share it) but favours c10 over c24. Why the ecalib mask is better (mid-frequency density, or
something else) was not isolated.

## Smooth vs hard sensitivity maps with CAIPI R=4 (2026-10-20, 3 validation volumes)
Script: `scripts/diagnostics/ecalib_smooth_vs_hard_caipi_r4.py`; outputs `reports/ecalib_smooth_vs_hard_caipi_r4/results.json`, `caipi_r4_errors.png`.
Fixed 2x2 CAIPI lattice plus the 10x10 centre (25.1% sampled, effective R = 3.98), CG-SENSE (10 iterations, lambda 0, no weights), so only the
maps differ. Volumes are the median-PSNR validation volume per class (by the conv run): brain `file_brain_AXT2_203_2030352` (14 slices), knee-PD
`file1001236` (33), knee-PDFS `file1001475` (32). Trainer-path metrics, all slices. ecalib_hard = the cached ecalib maps made net-style
(support sum|S|^2 > 0.5, unit norm inside, zero outside).

| Volume | net conv | net bilinear | ecalib soft | ecalib hard |
|---|---|---|---|---|
| brain PSNR / SSIM | 41.31 / 0.9851 | 41.44 / 0.9853 | 41.07 / 0.9833 | 41.24 / 0.9837 |
| knee-PD PSNR / SSIM | 42.96 / 0.9900 | 43.01 / 0.9899 | 46.30 / 0.9949 | 46.51 / 0.9950 |
| knee-PDFS PSNR / SSIM | 38.84 / 0.9146 | 38.79 / 0.9146 | 38.79 / 0.9236 | 38.81 / 0.9207 |

Paired per-slice PSNR, soft - hard ecalib: brain -0.167 dB (se 0.020), knee-PD -0.208 (0.048), knee-PDFS -0.026 (0.024). ecalib_hard - net_bilinear:
brain -0.20 (0.13), knee-PD +3.50 (0.53; 27/33 slices), knee-PDFS +0.02 (0.13).
- Hard-edging the ecalib maps does not hurt (PSNR slightly better, SSIM 0.003 lower on PDFS), so edge softness is not what separates ecalib from the net maps.
- The knee-PD gap to the net maps (+3.5 dB) is in the map estimate: the net's error shows aliasing streaks and a bright rim at the support edge
  (`caipi_r4_errors.png`); the 0.5% of signal pixels outside the net's support does not by itself account for 3.5 dB (not tested further).
- The ecalib maps use a 24x24 calibration of the fully sampled data while the CAIPI mask acquires only the 10x10 centre, so the ecalib arms use
  k-space that was not sampled; part of their advantage is this leak. Three volumes, one mask, one scoring path; volume-level conclusions only.

## Conclusions
- The early border blobs appear with transposed-conv upsampling and are absent with bilinear upsampling in both seeds
  (visual, one batch per epoch).
- The artefact does not affect the end result: epoch-50 validation PSNR/SSIM agree to 0.01 dB / 0.0002 in both seeds,
  the same as the seed-to-seed spread within an arm.
- Bilinear trains faster in the first epochs (+0.3 dB at epoch 1 in both seeds, mostly gone by epoch 5). Consistent
  with the blobs wasting early sampling budget; Hypothesis, untested.
- Slice-level differences between conv and bilinear are about as large as between two seeds of the same arm, so they
  are mostly training variation, not an upsampling effect. A small repeatable part exists (84% sign agreement on the 31
  slices above 0.25 dB in both seeds, mainly knee volume `file1001856`).
- That the blobs come from the transposed-conv checkerboard/ripple pattern is the original attribution;
  the matched comparison is consistent with it but does not isolate the mechanism (bilinear changes interpolation
  only, but a conv after upsampling was not tested here).

## Caveats
- Two seeds per arm. The final gap (0.01 dB) is not distinguishable from the seed-to-seed spread, but two seeds give
  only a rough spread, and no confidence intervals were computed.
- Pairs with the same seed share initialisation and data order, so the same-seed upsampling contrasts are partly
  correlated for reasons other than upsampling; the cross-seed sign agreement does not have that issue.
- Slice-level statistics come from 4 Bernoulli draws per slice and 10 knee volumes; the knee mean is dominated by
  one volume. The sections on sense maps and mask/sense-net swaps (seed 42 only) were not repeated for seed 43.
- The mask comparison is by eye on one training batch per epoch, sampled at selected epochs only (1, 2, 3, 5, 10, 20,
  50); epochs 4, 6-9 etc. were not inspected. No quantitative border-energy measure.
- The conv run was trained on the code as of 2026-10-17 and the bilinear run on later uncommitted code; configs
  match apart from the options noted above, but the old code was not re-run.
- CG-SENSE only. A separate 15-epoch VarNet bilinear run (`checkpoints/sens_upsample/loupe2d_r8_bilinear_15ep`,
  brain 36.13 dB / 0.930, knee 34.06 / 0.951) has no matched conv baseline and is not part of this comparison.
- These runs use the step lr schedule; the existing `*_bilinear_const_100ep` runs use a constant schedule and
  100 epochs and are not comparable.

## Figures
- `reports/sens_upsample_r8/mask_montage.png` (seed 42) and `mask_montage_seed43.png`: probabilistic mask at epochs 1, 2, 3, 5, 10, 20, 50; top conv, bottom bilinear.
- `reports/sens_upsample_r8/val_metrics.json` and `val_metrics_seed43.json`: per-epoch validation NMSE/PSNR/SSIM for both runs.
- `reports/sens_upsample_r8/ecalib_comparison_masks.png` (masks, differences, radial profiles), `ecalib_comparison_masks_draws.png` (p(k) and one Bernoulli draw), `ecalib_comparison_sense_maps.png` (sense maps on one median-PSNR mid-volume brain slice `file_brain_AXT2_201_2010588` slice 9 and knee slice `file1000495` slice 19).
- Per-run `metrics.png` and per-epoch training visualisations: `checkpoints/<run>/visualizations/`.

## Next steps
1. Quantify border energy: log mean p(k) over a border band per epoch from the sampler logits (the rendered
   figures are only colour-mapped images), for conv, bilinear, and bilinear + 3x3 conv.
2. Run the same pair with VarNet at 50 epochs to check the artefact and the early speed-up carry over.
3. Adopt bilinear as the default for new CG-SENSE runs: it removes the early artefact and speeds the first epochs at no
   cost to final quality in two seeds.
4. Look at why `file1001856` gains in both seeds (the repeatable part of the slice-level effect).
