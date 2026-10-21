# Can the 24 x 24 ACS choose a per-volume sampling mask for CG-SENSE at R=8? (2026-10-20)

## Question
The target pulse sequence (3D SPACE) acquires a fully sampled calibration centre (ACS) first, and the sampling
pattern for the rest of the scan may depend on it. The pattern is then fixed for the whole volume, because every
readout position shares the phase-encode mask. Is there per-volume mask adaptation worth building as ACS -> mask?
Two approaches were tested first: static masks per scan protocol, and a deterministic decoupled-comparison sampler
(DCM, Sun et al., IEEE TCI 2024). Then a pre-registered gate measured the per-volume ceiling on the LOUPE path.

Conventions:
- R=8: 12.5% of the 256 x 256 k-space, centre included.
- PSNR is the trainer's background-masked metric (`_apply_signal_mask`, data range = slice max), in dB.
- "Ring" means distance from the k-space centre in pixels.
- 95% intervals resample volumes (2000 bootstraps).

## Runs compared
Common to all runs:
- **Model:** CG-SENSE (10 CG iterations, lambda 0), so the only learned parts are the mask and, unless stated, the
  sensitivity U-Net. The sampler is `loupe_2d` with straight-through gradients, slope 10.
- **Training:** 50 epochs, lr 1e-3 stepped to 1e-4 at epoch 40, batch 16, L1 loss, seed 42.
- **Data:** fastMRI brain T2 + knee, balanced, 256 x 256. Brain volumes have 4-20 coils; the loader pads or crops
  to 16. Train has 1350 slices per anatomy
  (89 brain / 40 knee volumes); validation has 344 per anatomy (23 / 10 volumes).
- **Validation numbers:** the trainer's final-epoch validation (training log, n = 344 per anatomy).

| Group | wandb | Checkpoint (`checkpoints/`) | Sense maps | Preselected centre | Selection | Masks | Val brain PSNR / SSIM | Val knee PSNR / SSIM |
|---|---|---|---|---|---|---|---|---|
| LOUPE shared | xb6tna6g | `loupe2d_cgsense_r8_brain_knee_shared` | learned U-Net | 10 x 10 | Bernoulli | 1 | 34.92 / 0.913 | 32.70 / 0.929 |
| LOUPE per-anatomy | tk3c5ry8 | `..._anatcond` | learned U-Net | 10 x 10 | Bernoulli | brain, knee | 35.05 / 0.915 | 32.54 / 0.926 |
| LOUPE per-protocol | 3dbidc2q | `..._prototable` | learned U-Net | 10 x 10 | Bernoulli | brain, knee-PD, knee-PDFS | 35.04 / 0.915 | 32.41 / 0.924 |
| DCM shared | bz85wcp8 | `..._dcm_shared` | learned U-Net | 10 x 10 | DCM (p 1, tau 5) | 1 | 34.43 / 0.908 | 32.42 / 0.924 |
| DCM per-anatomy | gtzmmr4w | `..._dcm_anatcond` | learned U-Net | 10 x 10 | DCM | 2 | 34.67 / 0.911 | 32.23 / 0.920 |
| DCM per-protocol | mniw8jyy | `..._dcm_prototable` | learned U-Net | 10 x 10 | DCM | 3 | 34.64 / 0.911 | 32.20 / 0.920 |
| ecalib, 24 x 24 centre (gate reference) | ijs8t3rh | `..._ecalib_c24` | BART ecalib from 24 x 24 ACS | 24 x 24 | Bernoulli | 1 | 35.30 / 0.927 | 33.25 / 0.938 |
| ecalib, 10 x 10 centre | ltb2crlo | `..._ecalib_c10` | BART ecalib from 24 x 24 ACS | 10 x 10 | Bernoulli | 1 | 35.28 / 0.927 | 33.30 / 0.939 |

All models are healthy (SSIM 0.91-0.94). Every row is a single seed.

## Methods
1. **Mask variants (protocol, DCM).**
   - `scripts/eval/eval_cgsense_mask_variants_r8.py` scores the six learned-net runs on the full validation set,
     each with its own masks. It averages two seeded mask draws and reports paired per-slice differences against
     LOUPE shared.
   - **Class swap:** each slice is also scored with another class's mask (brain <-> knee; knee-PD <-> knee-PDFS).
   - DCM (`sampler.selection=dcm`): mask = top-M of (radial prior ((D - r) / D)^p minus the learned rejection map
     rescaled to the prior's mean). The backward pass is straight-through via M softmax(score / tau).
2. **DCM training check.** `scripts/diagnostics/dcm_training_check_r8.py` reports:
   - the share of available locations whose score lies within 0.01 of the top-M threshold;
   - Dice of the mask between `best_model.pt` and `last_model.pt`;
   - how far the rejection logits moved from their seeded initialisation.

   `scripts/diagnostics/mask_variants_figure_r8.py` gives each mask's neighbour pair correlation g(1). g(d) is the
   mean of p(k) p(k+d) over its smoothed counterpart; below 1 means neighbours are avoided.
3. **Ecalib population models.** Missing BART ecalib maps (24 x 24 calibration, the repo's tuned presets) were
   added to the sense cache with `scripts/data/build_sense_cache.py`. With `datasets.sense_cache_dir` set, the
   model uses those maps instead of the U-Net, so the mask is the only learned part. Two runs differ only in
   `sampler.preselect_num` (24 or 10). `eval_cgsense_mask_variants_r8.py --runs ecalib_c10 ecalib_c24` gives the
   paired difference.
4. **The gate** (`scripts/diagnostics/acs_mask_gate_cgsense_r8.py`, launched by `analysis/acs_gate_cgsense/run_gate*.sh`).
   - **Split:** each validation volume's slices, in order, are split into F (first half) and H (second half, after
     a 2-slice gap). Up to 16 slices of each are used.
   - **Fitting:** per-volume masks are fitted through the frozen ecalib 24 x 24 model with Adam on the L1 training
     loss (4 slices x 2 Bernoulli draws per step). Scoring is on H with 4 seeded draws shared by every arm.
   - **Arms:**
     - `mask`: a free logit map;
     - `mask_s4`, `mask_s8`: that map kept above 4 or 8 px, i.e. the population probability times the map's
       Gaussian-smoothed density ratio, which keeps every ring's budget;
     - `grid`: the population density times exp(bicubic 8 x 8 grid theta), a smooth change an ACS predictor could
       output.
   - **Companions:** each arm has a `_ctrl`, the same fit on as many training slices from other subjects. It
     captures refit gain that is not specific to the volume. Each also has an `_in`, fitted on F and H: the
     in-sample oracle, which must not lose to the population mask.
   - **Rule (in the docstring before any run):** stop unless, in at least one anatomy, a smooth held-out arm
     (`grid`, `mask_s4`, `mask_s8`) beats its control by >= +0.05 dB with the interval above 0. A stage-2 ACS
     predictor would then have to beat a static refit by >= +0.1 dB, and coil-wise ACS features would have to beat
     coil-combined ones.
   - **Runs:** run 1 used free lr 0.05, grid lr 0.02 and 300 steps. Run v2 (`--tag v2`) used free lr 0.005,
     grid lr 0.002 and 600 steps.
5. **Fit check.** `scripts/diagnostics/acs_gate_fit_check_r8.py` refits the in-sample arms on 3 knee and 2 brain
   volumes under several step sizes and objectives. Every 25-50 steps it records the full-image L1 and the masked
   PSNR on the fitted slices. This separates a fit that fails to lower its loss from a loss that disagrees with the
   metric.
6. **Theta check.** `scripts/diagnostics/acs_gate_theta_check_r8.py --direction pc1|gain` uses the v2 brain thetas.
   - **Direction:** the effective change r = log(p / p_base) after budget rescaling. Similarity between volumes is
     the cosine of r weighted by p_base.
   - **Decomposition:** each volume's raw F-fit theta (`full_fit`, which reproduces the gate) is split into its mean
     (`fit_offset`) and the rest (`fit_centred`).
   - **One-direction arms, leave one volume out:** u is either PC1 of the other volumes' thetas (`pc1`) or their
     average weighted by in-sample gain (`gain`).
     - `dir_oracle` uses the volume's own projection onto u.
     - `dir_static` uses the others' mean amplitude.
     - `dir_<scalar>` uses an amplitude linearly predicted from one per-volume ACS scalar, computed from the
       24 x 24 centre only:
       - `acs_spread`: energy share outside the central 8 x 8;
       - `acs_coils`: coil participation ratio (sum E_c)^2 / sum E_c^2;
       - `acs_object`: share of the low-resolution ACS image above 10% of its max.
     - `base_psnr` is a non-deployable reference.
   - **Coil count:** the same script records each volume's coil count (non-zero coils after the loader) and splits
     the held-out gain at 8 coils.
7. **Scanner check.** `scripts/diagnostics/acs_gate_scanner_check_r8.py` reads the ISMRMRD header of every brain
   volume, train and val: system model, field strength, receiver channels and raw k-space coil count. It joins these
   with the held-out gains (output `analysis/acs_gate_cgsense/scanner_check.json`).
8. **Figure.** `scripts/diagnostics/plot_acs_gate_r8.py`.

## Results

### Static per-protocol masks and DCM do not beat one shared LOUPE mask
Paired per-slice PSNR difference vs LOUPE shared (dB), full validation set:

| Run | Brain | Knee | Knee-PD | Knee-PDFS |
|---|---|---|---|---|
| LOUPE per-anatomy | +0.101 [+0.039, +0.185] | -0.145 [-0.229, -0.082] | -0.118 | -0.163 |
| LOUPE per-protocol | +0.089 [+0.027, +0.180] | -0.433 [-0.485, -0.366] | -0.407 | -0.449 |
| DCM shared | -0.846 [-0.985, -0.696] | -0.609 [-0.712, -0.475] | -0.739 | -0.526 |
| DCM per-anatomy | -0.534 [-0.659, -0.374] | -0.870 [-0.997, -0.716] | -1.050 | -0.755 |
| DCM per-protocol | -0.551 [-0.663, -0.405] | -0.956 [-1.067, -0.790] | -1.018 | -0.915 |

Per-protocol masks gain the same on brain as per-anatomy masks and lose more on knee. A knee slice scores
+0.058 [-0.049, +0.188] dB better with its own protocol's mask than with the other knee protocol's, so there is no
protocol-specific mask to learn. DCM loses
0.5-1.05 dB everywhere.

| DCM diagnostic | DCM runs | LOUPE runs |
|---|---|---|
| Available locations within 0.01 of the top-M threshold (shared run) | 74% (6.0x the budget) | n/a |
| Dice of the mask, best vs last checkpoint | 0.21-0.30 | n/a |
| Correlation of trained with initial rejection map | 0.03-0.07 (knee-PD 0.34) | 0.38-0.46 |
| Neighbour pair correlation g(1) | 0.99-1.18 | 0.39-0.54 |
| Share of samples beyond 64 px | 0.57-0.68 | 0.56-0.69 |

The DCM ordering is near-tied: most locations sit at the threshold, so the chosen set is decided by tiny score
differences and keeps changing. This implementation learned no stable mask and lost the spacing that LOUPE learns.
These DCM numbers describe this implementation (static rejection map, p 1, tau 5), not DCM in general.

### The 24 x 24 centre costs 0.08-0.13 dB
`ecalib_c24` minus `ecalib_c10`, same maps: brain -0.079 [-0.105, -0.042], knee -0.133 [-0.174, -0.092]
(PD -0.191, PDFS -0.096). This is the sampling cost of fixing 476 more samples in the centre. Both runs use maps
from 24 x 24 data. At the same 10 x 10 centre, ecalib maps put the population model 0.36 dB (brain) and 0.60 dB
(knee) above the learned-U-Net shared run on trainer validation (35.28 vs 34.92; 33.30 vs 32.70).

### Gate run 1 is invalid; v2 passes on brain and stops on knee
Gain in dB (95% interval):

| Arm | Run 1, brain | Run 1, knee | v2, brain | v2, knee |
|---|---|---|---|---|
| `grid` - `grid_ctrl` (held out) | +0.213 [+0.088, +0.354] | -0.242 [-0.407, -0.107] | **+0.157 [+0.068, +0.258]** | -0.053 [-0.120, +0.010] |
| `mask_s8` - `mask_s8_ctrl` | +0.214 [+0.017, +0.452] | -0.282 [-0.462, -0.128] | +0.167 [+0.066, +0.301] | -0.057 [-0.138, +0.009] |
| `mask` - `mask_ctrl` | +0.292 [+0.090, +0.529] | -0.305 [-0.461, -0.159] | +0.182 [+0.081, +0.305] | -0.055 [-0.128, +0.011] |
| `grid_in` - base (in-sample) | +0.202 [+0.086, +0.332] | **-0.107 [-0.142, -0.075]** | +0.155 [+0.073, +0.245] | -0.060 [-0.086, -0.034] |
| `mask_in` - base (in-sample) | +0.258 [+0.091, +0.454] | -0.087 [-0.142, -0.030] | +0.224 [+0.139, +0.327] | +0.010 [-0.016, +0.036] |
| `mask_ctrl` - base | -0.299 [-0.405, -0.190] | -0.237 [-0.331, -0.149] | -0.015 [-0.063, +0.029] | -0.078 [-0.142, -0.029] |
| `grid_ctrl` - base | -0.002 [-0.053, +0.047] | -0.147 [-0.237, -0.082] | +0.003 [-0.041, +0.046] | -0.134 [-0.240, -0.065] |

Run 1 failed its own sanity check: knee in-sample fits lost to the population mask, and every free-mask control lost
0.21-0.43 dB. Its brain "pass" is therefore not trusted.

Fit check (in-sample, change vs the population mask on the fitted slices):

| Volume | grid, lr 0.02, 300 steps (run 1) | grid, lr 0.002, 300 / 600 steps (v2: 600) | grid, PSNR objective, lr 0.02, 300 steps | free, lr 0.05 (run 1 lr), 600 steps | free, lr 0.005, 600 steps (v2) |
|---|---|---|---|---|---|
| knee 1000662 | L1 +0.21%, -0.020 dB | +0.035 / +0.011 dB | L1 +0.58%, -0.123 dB | L1 -0.16%, +0.204 dB | L1 -0.21%, +0.131 dB |
| knee 1000401 | L1 +0.27%, -0.084 dB | +0.005 / -0.002 dB | L1 +0.51%, -0.115 dB | L1 -0.03%, -0.163 dB | L1 -0.10%, +0.053 dB |
| knee 1001437 | L1 +0.20%, -0.043 dB | -0.014 / -0.028 dB | L1 +0.22%, -0.049 dB | L1 -0.09%, -0.016 dB | L1 -0.17%, +0.043 dB |
| brain 2010571 | L1 -9.85%, +0.990 dB | +0.723 / +0.812 dB | L1 -9.38%, +0.943 dB | L1 -14.23%, +1.491 dB | L1 -9.63%, +0.928 dB |
| brain 2080126 | L1 -3.72%, +0.796 dB | +0.546 / +0.651 dB | L1 -3.52%, +0.757 dB | L1 -5.04%, +1.120 dB | L1 -3.44%, +0.769 dB |

On knee no setting lowers the grid's L1 even in-sample, and fitting PSNR directly fails the same way. So the
objective agrees with the metric. With large steps, Adam's normalised updates random-walk the mask away from the
population optimum. The v2 settings were chosen from these in-sample columns only, and keep every in-sample free
fit at or above base.

v2 brain concentration (smooth arm minus control):

| Arm | Median | Positive | Top-5 share of the sum | Corr(base PSNR, gain) | Without the ~25 dB outlier 2010571 and the 1-slice volume |
|---|---|---|---|---|---|
| grid | +0.063 | 19/23 | 0.75 | -0.31 | +0.091 [+0.044, +0.148] |
| mask_s4 | +0.030 | 19/23 | 0.84 | -0.33 | +0.092 [+0.032, +0.169] |
| mask_s8 | +0.041 | 19/23 | 0.83 | -0.33 | +0.089 [+0.032, +0.153] |

Brain passes the pre-registered rule on all three smooth arms, with controls at about 0 and in-sample about equal to
held-out. Three-quarters of the gain comes from five volumes. Knee stops.

### The gaining volumes want one change, but no ACS scalar sets its size
v2 brain, held-out gain vs the population mask (`dir_static` is the same amount for every volume):

| Arm | `--direction pc1` | `--direction gain` |
|---|---|---|
| full_fit (own F-fit theta) | +0.160 [+0.075, +0.254] | same |
| fit_centred / fit_offset | +0.112 [+0.055, +0.177] / -0.015 [-0.031, -0.002] | same |
| dir_oracle (own amplitude) | -0.007 [-0.026, +0.011] | +0.072 [-0.013, +0.173] |
| dir_static | -0.000 [-0.018, +0.016] | +0.016 [-0.045, +0.078] |
| dir_acs_coils | -0.002 [-0.020, +0.014] | +0.050 [-0.010, +0.120] |
| dir_acs_object | +0.000 [-0.018, +0.017] | +0.030 [-0.038, +0.099] |
| dir_acs_spread | -0.001 [-0.019, +0.017] | +0.021 [-0.034, +0.087] |
| dir_base_psnr (not deployable) | -0.003 [-0.020, +0.014] | +0.028 [-0.031, +0.091] |

- **Similarity:** the top-5 gainers' changes have weighted cosine 0.91 with each other, against 0.20 over all pairs.
  A volume's F fit matches its F+H fit at 0.92, and its control at 0.21. PC1 of all thetas explains 63% of their
  variance, but its amplitude does not track gain (r = -0.07, oracle -0.007 dB). The gain-weighted direction does
  (r = 0.84).
- **Offset and shape act together.** The population map's mean probability over available locations (0.138) sits
  just above the rescale target (0.117). The top-5 volumes' fitted thetas have means of -0.16 to -0.20 (mean of
  `grid_theta['grid']` in `analysis/acs_gate_cgsense/brain_v2/*.json`), against about 0 for most others. The gainers' theta offset therefore switches the
  budget rescale into its inverted branch, which adds a uniform floor. The offset alone loses 0.015 dB and the
  centred shape keeps 70% of the gain.
- **No scalar reaches the bar.** The best one is `acs_coils`, at +0.050 [-0.010, +0.120] against the population
  mask, which is also the static refit (`grid_ctrl` +0.003). Even a perfect amplitude gives +0.072, below the
  +0.1 dB stage-2 bar.

### The gain follows the few-coil (mostly Avanto 1.5 T) volumes
| Volumes (brain, v2) | n | Held-out grid gain over control |
|---|---|---|
| 4-6 coils | 5 | +0.408 [+0.234, +0.638], 5/5 positive |
| 14-20 coils (the loader crops 20 to 16) | 18 | +0.087 [+0.024, +0.188], 14/18 positive (includes the outlier 2010571, +0.83) |

Gain vs coil count: r = -0.52 (`acs_gate_theta_check_r8.py`, coils counted after the loader).

Across the 5 few-coil volumes, the in-sample fit moves 5.8% of the budget out of the ring beyond 64 px, into
32-64 px (+4.5%) and 16-32 px (+1.2%). The 14-20-coil volumes change by under 1% in any ring.

Scanner headers (`acs_gate_scanner_check_r8.py`):

| System model | Field (T) | Coils | Val volumes | Mean held-out gain |
|---|---|---|---|---|
| Avanto | 1.5 | 4-6 | 4 | +0.468 |
| TrioTim | 3 | 4 | 1 | +0.172 |
| Aera | 1.5 | 16-20 | 3 | +0.016 |
| Biograph_mMR | 3 | 14 | 3 | +0.002 |
| Prisma_fit | 3 | 20 | 1 | +0.101 |
| Skyra | 3 | 16-20 | 11 | +0.128 (includes 2010571, +0.83) |

Field strength does not explain the gain: the many-coil 1.5 T Aera volumes gain about 0, while the 1.5 T Avanto
volumes gain about +0.47. Coil count cannot be separated from scanner model, however: every Avanto volume has 4-6
coils, and only one few-coil volume (TrioTim) comes from another model. In training, 22 of 89 brain volumes have at
most 8 coils (17 Avanto with 4 coils, 2 Avanto with 6, 2 TrioTim with 4, 1 Skyra with 8).

## Conclusions
- **Static routing beyond anatomy does not help.** Per-protocol masks lose 0.43 dB on knee, and the PD and PDFS
  masks are interchangeable (Results, first table).
- **This DCM implementation does not train a mask.** About 74% of locations tie at the threshold and the mask
  churns between checkpoints, costing 0.5-1.05 dB. Any rejection-map design on this path would need stochastic or
  annealed top-M first.
- **Per-volume mask adaptation has a real brain ceiling on CG-SENSE: +0.16 dB held out over a matched control.**
  It is concentrated in few volumes (median +0.06; +0.09 without the outlier) and absent on knee.
- **The ACS -> mask line is a no-go at stage 2.**
  - The gaining volumes want one shared change: fewer samples beyond 64 px, more in 16-64 px.
  - With a perfect amplitude that change is worth +0.072 dB; the best 24 x 24 ACS scalar gets +0.050 dB.
  - Both are below the pre-registered +0.1 dB.
- **The adaptive signal is mostly the few-coil volumes.** Volumes with 4-6 coils gain +0.41 dB, the rest +0.09.
  - Field strength is ruled out, but coil count is confounded with scanner model (4 of the 5 are Avanto).
  - Either way the variable is known from the scan header, so this is a routing lever, not ACS adaptivity.
  - Hypothesis, untested: with few coils CG-SENSE is worse conditioned (g-factor) in outer k-space, so the optimal
    density is more central.
- **The larger ACS has a cost** of 0.08 dB on brain and 0.13 dB on knee. On a fixed coil array none of the
  adaptation measured here would offset it on knee.

## Caveats
- Single training seed for every population model. The gate is one fit per arm and volume, with 4 scoring draws.
- **Post-hoc fitting settings.** Run 1's settings were replaced after it failed its in-sample check. The v2 settings
  were picked on 5 of the gate's own validation volumes, using in-sample results only. Run 1's numbers are reported
  above for comparison.
- **Post-hoc analyses:**
  - The gain-weighted direction was introduced after the pre-registered PC1 reading passed on paper (63% of
    variance, top-5 cosine 0.91) but carried no gain.
  - The outlier exclusion (volume 2010571, a known ~25 dB brain validation outlier) and the coil-count split were
    chosen after looking.
  - None of these changes the stage-2 verdict.
- Knee `grid_in` is still -0.060 dB in-sample in v2, so the knee grid arm does not fully meet its own sanity check.
  The knee stop rests on the fit check, which finds no in-sample L1 decrease at any setting.
- Coil count and scanner model are confounded: 4 of the 5 few-coil validation volumes, and 19 of the 22 few-coil
  training volumes, are Avanto. Field strength is not the driver, but "few coils" and "Avanto" cannot be told apart
  with these data.
- Only 23 brain and 10 knee validation volumes, and only 5 few-coil volumes. The ACS scalars are three hand-picked
  features with a linear amplitude model; a learned ACS encoder was not tried.
- fastMRI brain T2 slices stand in for a 3D SPACE volume: a "volume" here is a stack of 2D slices sharing one mask.
- Both ecalib runs use maps from 24 x 24 calibration data, so the c24 vs c10 difference is the sampling cost only.
  The learned-U-Net vs ecalib comparison mixes map quality with the absence of a learned map network.

## Figures
- `reports/figures/acs_mask_gate_cgsense_r8.png`.
  - Left: held-out grid gain per volume, brain coloured by raw coil count (4-6 vs 14-20), knee in grey.
  - Right: change in the share of samples per ring (in-sample grid fit minus population mask), by coil count.
  - Generated by `scripts/diagnostics/plot_acs_gate_r8.py`.
- `reports/figures/cgsense_mask_variants_r8.png`: LOUPE vs DCM masks for the shared, per-anatomy and per-protocol
  runs, with the share beyond 64 px and g(1) (`scripts/diagnostics/mask_variants_figure_r8.py`).

Data:
- `analysis/cgsense_mask_variants/` (eval.md, dcm_training_check.json, masks.json);
- `analysis/acs_gate_cgsense/`:
  - summary.md, summary_v2.md;
  - population_c24_vs_c10.md;
  - fit_check_*.json;
  - theta_check_{pc1,gain}.json, scanner_check.json;
  - per-volume JSONs in brain/, knee/, brain_v2/, knee_v2/.

## Next steps
1. **Coil-count routing:** train the ecalib CG-SENSE LOUPE with `datasets.condition_classes` split by coil count
   (<= 8 vs > 8). Check whether the few-coil volumes keep their +0.4 dB with a static per-class mask (about
   1.5 h).
   - The few-coil class has 22 training volumes, about the size of the knee-PD class, whose own mask did not
     generalise.
   - Initialising the class masks from the shared mask, or fitting only an 8 x 8 grid per class, would limit that
     risk.
2. **ACS-chosen acceleration** (the parked budget-adaptation idea): the volumes that gain here are the low-PSNR,
   unusual ones, which is the population per-volume R adaptation targets.
3. If DCM is revisited, replace the deterministic top-M with a Gumbel-perturbed or tau-annealed top-M, and check that
   the rejection map stabilises (best vs last Dice) before any conditioning.
