# Can one sampling mask and one VarNet per subject recover the per-slice oracle gain at R=8? (2026-10-18)

## Question
A mask fitted to one slice's ground truth through a frozen VarNet beats the per-anatomy mask by about +0.9 dB
(conv VarNet, `reports/ADAPTIVITY_MECHANISMS_R8.md`). That mask cannot be predicted from the calibration centre, and
it does not transfer to other slices. Two questions follow:
- Does co-adapting the reconstructor as well (one mask **and** one VarNet per subject) recover that gain, at a
  granularity a scan can use?
- Can a per-scan VarNet be adapted without ground truth, by zero-shot self-supervised fine-tuning (SSDU or ECHO)?

A SPACE scan uses one phase-encode mask per volume, so "per subject" means one mask per validation volume.

Conventions:
- R=8 point-wise 2D masks, with the 10 x 10 centre preselected.
- PSNR is the trainer's background-masked metric (`_apply_signal_mask`), in dB, paired per slice.
- Brackets are 95% CIs from bootstrapping volumes. Means are slice-weighted.

## Runs compared
Every arm starts from the same model.

| Model | Checkpoint | wandb | Settings | Val PSNR / SSIM brain | Val PSNR / SSIM knee |
|---|---|---|---|---|---|
| LOUPE VarNet (base) | `checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05/last_model.pt` | `uo8d68n0` | loupe_2d slope 5, VarNet 12 cascades, `dc_mode=where`, brain_knee_balanced, R=8, 50 epochs, seed 42 | 36.55 / 0.933 | 34.26 / 0.953 |

Validation set: 23 brain volumes (344 slices) and 10 knee volumes (344 slices).

## Methods
1. **Ground-truth ceiling** (`scripts/diagnostics/subject_coadapt_ceiling_r8.py`). Each validation volume's slices
   are split into a fit set F and a scored set H, so a fitted model never sees an H image.
   - **Alternating split:** F = even slices, H = odd slices. Every scored slice is adjacent to a fitted one.
   - **Block split:** F = the first half, H = the second half after a 2-slice gap. The nearest fitted slice is 3-8
     slices away (brain) and 3-22 slices away (knee). This is the subject-level test.

   Up to 16 F and 8 H slices per volume. H is scored with 4 seeded Bernoulli draws per slice, the same draws for
   every arm. Arms:

   | Arm | What is fitted (ground-truth L1 loss) | On which slices |
   |---|---|---|
   | `mask` | one free logit map per volume, VarNet frozen (Adam lr 0.05, init = LOUPE logits, 150 steps) | F |
   | `mask_in` | the same, as the per-volume mask oracle a SPACE scan could use (150, 600 or 1500 steps) | F and H |
   | `mask` at 1500 steps | the transfer test of the long oracle: fitted like `mask_in`, but on F only | F |
   | `recon` | VarNet + sensitivity net, LOUPE mask (Adam lr 1e-4, 50-step warmup, clip 1.0) | F |
   | `joint` | mask and VarNet together, same optimisers | F |
   | `recon_on_mask_in` | VarNet + sensitivity net, with the 600-step `mask_in` map held fixed | F |
   | `recon_generic` | VarNet + sensitivity net: plain continued training | random slices of the whole training set |
   | `recon_ctrl`, `mask_ctrl`, `joint_ctrl` | the same fit on as many training slices from other subjects | |

   Batches are 4 slices x 2 draws. Fine-tuned arms are scored at 25 (block split only), 50, 100, 200 and 400 steps.
   *Typical* slices exclude those with more than 30% of the target's signal outside the sense support; this is the
   mask-independent tail.
2. **Optimiser check.** On one brain and one knee volume (block split), `recon` and `recon_generic` are run twice:
   - with a fresh Adam;
   - with Adam resuming the checkpoint's own moments (`--resume-adam`).
3. **Self-supervised per-scan adaptation** (`scripts/diagnostics/subject_ssl_tta_r8.py`). Same base model, sensitivity
   net frozen unless stated. Each slice is acquired once (seeded draw 0, the acquired set Omega). The VarNet is
   fine-tuned on acquired k-space only and scored at the same draw, with all of Omega as input.
   - **Per slice:** each H slice gets its own fit, on two H slices of brain volume `AXT2_204_2040011`.
   - **Per volume:** one fit uses every slice of the volume and is scored on its H slices, on brain `AXT2_204_2040011`
     and knee `file1001856`.

   Methods:
   - **SSDU / ZS-SSL** (Yaman et al.): each step splits Omega into an input Theta and a loss set Lambda (`rho` = 0.4
     of the non-centre points). The loss is SSDU's normalised L1 + L2 on Lambda, computed either on the image
     re-projected through the sense maps (`image`) or on the VarNet's per-coil k-space (`coil`).
   - **ECHO** (Shang et al.): the current VarNet, without gradients, estimates the full k-space from Omega. The
     acquired values replace the estimate on Omega, giving the hybrid. A fresh LOUPE draw (same R) re-undersamples
     the hybrid as the input. The loss is per-coil against the acquired data, on one of:
     - all of Omega (`acquired`);
     - the Omega points the new mask left out (`unseen`);
     - a held-out share Lambda (`rho` = 0.2), also removed from the estimator's input and from the hybrid
       (`heldout`, no leakage).
4. **Density-mismatch diagnostic** (`scripts/diagnostics/ssl_density_mismatch_r8.py`). Brain validation slice 20,
   SSDU coil loss, lr 1e-5, 50 steps, sensitivity net frozen. At each step it records:
   - the ground-truth PSNR with Omega as input (deployed) and with a fixed Theta as input (trained);
   - the re-projection error of fft(x S) on Theta and on Lambda.

All tables: `scripts/diagnostics/summarize_subject_coadapt_r8.py` -> `analysis/subject_coadapt/summary.md`.

## Results

### Subject level (block split): brain 16 volumes / 95 slices, knee 6 volumes / 48 slices
| Arm vs base (dB) | Brain | Knee |
|---|---|---|
| `mask_in` (150 steps), per-volume mask oracle, frozen VarNet | +0.218 [+0.160, +0.272] | +0.165 [+0.052, +0.244] |
| `recon` @25 | +0.021 [+0.010, +0.033] | -0.018 [-0.060, +0.017] |
| `recon` @50 | +0.016 [-0.001, +0.035] | -0.006 [-0.054, +0.043] |
| `recon` - `recon_generic` @50 (subject vs continued training) | +0.035 [+0.024, +0.049] | +0.024 [-0.041, +0.078] |
| `joint` @25 | +0.045 [+0.023, +0.068] | -0.012 [-0.040, +0.016] |
| `joint` - `recon` @25 (mask gain on a co-adapted VarNet) | +0.024 [+0.008, +0.043] | +0.006 [-0.026, +0.041] |
| `recon` @400 | -0.387 [-0.492, -0.283] | -0.263 [-0.373, -0.192] |
| `joint` @400 | -0.484 [-0.610, -0.348] | -0.436 [-0.505, -0.357] |

Fitting a VarNet, or a VarNet and a mask, to a subject's other slices gains at most +0.05 dB at 25-50 steps.
Beyond that it overfits the fitted half and loses 0.3-0.5 dB. The only subject-level gain of any size is the
in-sample per-volume mask, which is fitted with the volume's own ground truth.

### Per-volume mask oracle, and a VarNet co-adapted to it (block split, brain 4 / knee 3 volumes, 24 slices each)
`joint` fits its mask on F only, and an F-fitted mask hurts H (next table). So the clean test of co-adaptation holds
`mask_in` fixed and fine-tunes the VarNet on F (`recon_on_mask_in`). Amplification would show up as the VarNet gaining
more on the oracle mask than on the LOUPE mask.

| PSNR (dB) | Brain | Knee |
|---|---|---|
| `mask_in` 600 steps - base | +0.384 [+0.167, +0.558] | +0.213 [-0.032, +0.436] |
| `mask_in` 1500 steps - base | +0.527 [+0.335, +0.704] | +0.323 [+0.080, +0.577] |
| `recon` - base @25 / @50 / @100 (VarNet gain on the LOUPE mask) | +0.021 / +0.012 / -0.035 | +0.017 / +0.113 / +0.077 |
| `recon_on_mask_in` - `mask_in` @25 / @50 / @100 (VarNet gain on the oracle mask) | +0.024 / +0.035 / +0.001 | +0.057 / +0.150 / +0.149 |
| `recon_on_mask_in` - base @50 | +0.419 [+0.210, +0.608] | +0.363 [+0.047, +0.579] |

- **The gains add; the VarNet barely amplifies the mask.**
  - On brain, the VarNet adds the same +0.00-0.04 dB on the oracle mask as on LOUPE's.
  - On knee it adds +0.04-0.07 dB more on the oracle mask, with wide intervals (3 volumes).
- **The oracle keeps rising in-sample but does not transfer within the volume.** Its map's correlation with LOUPE's
  falls from 0.79 to 0.42. Same volumes, block split, PSNR vs base in dB:

  | Fit steps | `mask_in`, scored on the fitted slices: brain / knee | `mask`, fitted on F and scored on H: brain / knee |
  |---|---|---|
  | 150 | +0.242 / +0.052 (2 of 3 knee volumes) | |
  | 600 | +0.384 / +0.213 | |
  | 1500 | +0.527 / +0.323 | -0.200 [-0.349, -0.022] / -0.198 [-0.266, -0.124] |

  A longer fit raises the in-sample number and makes the out-of-sample number worse. The per-volume oracle fits
  those exact slices, not the subject.

### Adjacent slices (alternating split): brain 12 volumes / 92 slices, knee 1 volume / 8 slices
| Arm vs base (dB) | Brain | Knee (1 volume) |
|---|---|---|
| `mask` (fitted on even slices, scored on odd ones) | -0.099 [-0.175, -0.022] | -0.027 |
| `mask_in` (150 steps) | +0.235 [+0.171, +0.289] | +0.124 |
| `recon` @50 | +0.129 [+0.091, +0.177] | +0.767 |
| `recon` @100 | +0.154 [+0.108, +0.216] | +0.882 |
| `recon` @400 | -0.034 [-0.106, +0.057] | +1.050 |
| `joint` @50 | +0.188 [+0.115, +0.290] | +0.817 |
| `joint` - `recon` @50 | +0.060 [+0.021, +0.114] | +0.050 |

- **Neighbouring slices make the gains larger:** brain +0.13 to +0.19 dB, and one knee volume +0.8 to +1.1 dB.
- **That knee volume under the block split:** +0.25 dB at 50 steps and −0.28 dB at 400. The alternating-split gain is
  mostly neighbouring-slice redundancy, which is per-slice information, not a subject-level prior.
- **A per-volume mask does not carry within a subject.** Fitted on half a volume's slices, it is worse than LOUPE on
  the other half (−0.10 dB), so the per-volume mask oracle is an in-sample effect.

### The fine-tune recipe is not the problem (block split, 1 volume each)
| PSNR vs base (dB) | @25 | @50 | @100 | @200 | @400 |
|---|---|---|---|---|---|
| brain `recon_generic`, fresh Adam | -0.021 | -0.013 | -0.058 | +0.030 | -0.035 |
| brain `recon_generic`, resumed Adam | -0.040 | -0.015 | -0.093 | +0.018 | -0.056 |
| knee `recon_generic`, fresh Adam | -0.061 | -0.169 | -0.170 | -0.060 | -0.173 |
| knee `recon_generic`, resumed Adam | -0.179 | -0.160 | -0.127 | -0.083 | -0.171 |
| brain `recon`, fresh / resumed Adam | +0.070 / +0.080 | +0.095 / +0.070 | +0.053 / +0.022 | -0.060 / -0.115 | -0.387 / -0.544 |
| knee `recon`, fresh / resumed Adam | +0.022 / +0.287 | +0.253 / +0.303 | +0.216 / +0.196 | +0.143 / -0.030 | -0.281 / -0.458 |

- **Plain continued training stays close to base:** −0.18 to +0.03 dB. Resuming the optimiser's moments changes
  little.
- **The knee volume shown (`file1001856`) is the best case for subject fitting:** about +0.4 dB over continued
  training at 25-50 steps. Across the other 6 knee volumes (`knee_block_full`) it averages +0.02 dB.
- **The other-subject controls are not baselines.** The matched-count `*_ctrl` arms lose 0.1-2.2 dB, because they
  specialise to 16-17 slices of other subjects.

### Self-supervised per-scan adaptation
Per slice (2 slices, brain `AXT2_204_2040011`; PSNR change vs base, dB):

| Method | Loss | lr | rho | Sense net | @10 | @25 | @50 | @100 | @200 | @500 | @1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ssdu | image | 1e-5 | 0.4 | trained | | -1.380 | -5.044 | -6.873 | -6.723 | | |
| ssdu | image | 3e-6 | 0.4 | frozen | -0.021 | -0.210 | -1.072 | -3.335 | | | |
| ssdu | coil | 3e-6 | 0.4 | frozen | -0.020 | -0.207 | -1.095 | -3.660 | | | |
| ssdu | coil | 1e-5 | 0.4 | frozen | -0.092 | -1.347 | -5.247 | -6.358 | | | |
| echo | acquired | 1e-5 | | frozen | | -0.317 | -0.297 | -0.374 | -0.437 | | |
| echo | acquired | 3e-5 | | frozen | | -0.378 | -0.421 | -0.423 | -0.456 | | |
| echo | acquired | 1e-4 | | frozen | | -0.557 | -0.466 | -0.518 | -0.585 | -1.178 | -2.097 |
| echo | acquired | 3e-4 | | frozen | | -0.609 | -0.592 | -0.685 | -0.988 | -2.065 | -3.444 |
| echo | unseen | 1e-5 | | frozen | | -0.199 | -0.140 | -0.207 | -0.290 | | |
| echo | unseen | 3e-5 | | frozen | | -0.121 | -0.258 | -0.309 | -0.386 | | |
| echo | heldout | 1e-5 | 0.2 | frozen | | -0.067 | -0.195 | -0.404 | -0.861 | -1.660 | |
| echo | heldout | 1e-4 | 0.2 | frozen | | -0.863 | -2.276 | -2.687 | -2.915 | -3.113 | |

Per volume (one VarNet fitted on every slice's acquired data; ECHO `unseen`, lr 1e-5; scored on H):

| Volume | @25 | @50 | @100 | @200 |
|---|---|---|---|---|
| brain `AXT2_204_2040011` | -0.221 | -0.082 | -0.200 | -0.285 |
| knee `file1001856` | -0.045 | -0.116 | -0.194 | -0.248 |

Every self-supervised variant loses PSNR against the supervised base, per slice and per volume, at every
learning rate and step count.
- **ECHO loses least.** Higher lr is worse, with no recovery after the early drop.
- **Leakage is not the cause.** The leak-free ECHO variant is worse than standard ECHO.
- **SSDU loses 1-7 dB** within 100 steps.
- **The loop itself is inert.** The same loop at lr 0 changes nothing (+0.000 dB).

### Why SSDU fails: it trains the VarNet for the wrong acceleration (brain slice 20, lr 1e-5, coil loss)
| Step | PSNR, Omega input (R=8, deployed) | PSNR, Theta input (R~13, trained) | Re-projection error on Theta / Lambda |
|---|---|---|---|
| 0 | 40.03 | 34.22 | 0.099 / 0.248 |
| 10 | 39.17 | 35.57 | 0.097 / 0.227 |
| 25 | 37.60 | 36.56 | 0.095 / 0.211 |
| 50 | 37.31 | 36.94 | 0.093 / 0.203 |

- **The self-supervised signal is real:** the trained-density reconstruction gains +2.7 dB against the ground truth
  in 50 steps.
- **But the VarNet becomes a specialist at the sparser input.** At the deployed R=8 it loses 2.7 dB.
- **ECHO** removes the density mismatch and loses far less, but still does not beat the supervised model.

## Conclusions
- **Co-adapting one mask and one VarNet per subject does not recover the ~1 dB.**
  - **Subject-level ceiling (block split):** a VarNet co-adapted to the subject adds +0.00 to +0.04 dB on brain, and
    +0.06 to +0.15 dB on knee (4 brain / 3 knee volumes), on top of a ground-truth per-volume mask.
  - **That is about what the same fine-tune adds on LOUPE's mask.** The gains add; there is little amplification.
  - **Subject fitting against plain continued training** gains +0.035 dB on brain (16 volumes) and +0.02 dB on
    knee (6 volumes). One further knee volume, `file1001856`, gains about +0.4. The best step is chosen with the ground truth, so even these are
    optimistic.
- **The ~1 dB is per-slice information.** It appears only when the fitted slices include neighbours of the scored
  ones (one knee volume, +1.05 dB). It is gone once fitted and scored slices are 3 or more slices apart.
- **The per-volume mask oracle is the only sizeable per-subject number, and it is purely in-sample.**
  - It reaches +0.53 dB (brain) and +0.32 dB (knee) at 1500 steps, and is still rising.
  - The same 1500-step fit, made on one half of the volume, scores −0.20 dB on the other half (brain and knee).
  - So it fits the specific slices it sees, much like the per-slice oracle. A predictor would have to reproduce
    slice-level detail, which the calibration-centre predictors could not.
  - Calibration-centre predictors of smooth per-slice masks recovered at most +0.03 dB
    (`reports/ADAPTIVITY_MECHANISMS_R8.md`).
- **Per-scan self-supervised adaptation does not beat this in-distribution supervised VarNet**, per slice or per
  volume.
  - ECHO: −0.14 to −0.46 dB per slice, −0.05 to −0.29 dB per volume.
  - SSDU fails through the train/inference acceleration mismatch.
- **Hypothesis, untested:** zero-shot adaptation would pay under distribution shift (another contrast, scanner or
  acceleration, e.g. SPACE data), which is where ECHO and ZS-SSL report their gains.

## Caveats
- **Volume counts:** the co-adaptation test (`recon_on_mask_in`) and the 600/1500-step oracles use 4 brain and 3 knee
  volumes. The alternating-split knee row is a single volume, because that shard OOM-crashed after it.
- **Single model and seed:** only the 50-epoch LOUPE VarNet (lr 1e-3) was tested. At this lr, longer training adds
  about 0.1-0.2 dB: the 100-epoch lr-1e-3 run gains +0.13 / +0.05 dB logged from epoch 50 to 100, and 10 more epochs
  on a 50-epoch base add +0.07 to +0.11 dB per slice. A better-converged base may leave a little less room for
  fine-tuning.
- **Untuned fine-tune:** the hyperparameters (lr 1e-4, batch 4 x 2, up to 400 steps) were not tuned. The best-step
  numbers are picked post hoc with the ground truth.
- **SSL scope:**
  - Per-slice results are 2 slices of one brain volume.
  - Per-volume ECHO covers one brain and one knee volume, at one setting.
  - The sensitivity net is frozen in all but the first SSDU row.
  - ECHO was applied as test-time adaptation of the pretrained 12-cascade VarNet, not as the paper's small network
    trained from scratch. ECHO always uses the per-coil loss.
- **Slices as proxy:** fastMRI 2D slices stand in for SPACE readout positions.

## Figures
None. Every table is in `analysis/subject_coadapt/summary.md`. Per-volume JSONs are under `analysis/subject_coadapt/`:
- `brain`, `knee`: alternating split;
- `*_block_full`, `knee_block`: block split;
- `*_block_maskin600`, `*_block_maskin1500`: the oracle follow-ups;
- `*_block_fresh`, `*_block_adam`: the optimiser check;
- `ssl/*`: self-supervised runs;
- `ssl_density/`: the density diagnostic.

Launchers are `analysis/subject_coadapt/run_*.sh`.

## Next steps
The scanner path is fixed as ACS -> pattern, one step. On this data nothing an ACS -> pattern mapping could output has
shown more than about 0.2 dB. Every larger number needs the slice's own ground truth, or fails on slices it was not
fitted to. The steps below decide whether any ACS lever exists; they are not a route to 1 dB.

1. **Held-out oracle over the pattern family a predictor would output** (reuses `subject_coadapt_ceiling_r8.py`,
   block split). **Done 2026-10-19: null by the rule below; see the update at the end.**
   - Replace the 65k free logits with a few per-volume parameters: density exponent, radial scale, anisotropy or
     ellipse aspect, optionally a coarse smooth basis.
   - Fit on F and score on H. Build an ACS -> parameters predictor only if the held-out gain is clearly positive
     (≥ 0.1 dB).
   - Expected: a null. The "same volume, other slices" smooth control in `reports/ADAPTIVITY_MECHANISMS_R8.md` was
     about −0.01 dB.
2. **Explain the anatomy-level sign flip.** This is the only ~0.2 dB effect, and anatomy or protocol routing is
   trivial on the scanner.
   - In the CG-SENSE runs (`xb6tna6g`/`tk3c5ry8` at 50 epochs, `u2xei0pq`/`kikc9627` at 100), the per-anatomy mask
     beats the shared one on brain (+0.13 / +0.16 dB) and loses on knee (−0.16 / −0.21).
   - A per-class mask can represent the shared mask, so losing on its own anatomy points to optimisation, the data
     split or a conditioning bug.
   - Test: initialise the per-anatomy masks from the shared mask, fine-tune, run a label swap, and add a second seed.
3. **Budget adaptation, if in scope: the ACS picks R per volume, not the shape.**
   - Per-volume base PSNR spans about 26-46 dB, far more than any mask effect.
   - The objective becomes scan time at matched quality, not dB at fixed R.
4. **Validate the metric before any SPACE-side evaluation.** SPACE has no ground truth, so patterns there can only be
   scored by held-out k-space loss, and that loss fell while ground-truth PSNR fell in the SSDU runs. On fastMRI,
   check first that held-out k-space loss ranks candidate masks as ground-truth PSNR does.
5. **A better proxy for SPACE.** SPACE masks are ky-kz per volume, which fastMRI 2D point masks may not represent.
   - Candidate fully sampled 3D Cartesian multi-coil sets: mridata.org 3D FSE knees, Calgary-Campinas multi-channel
     3D brain.
   - Verify that fully sampled ky-kz raw k-space is available before planning on them.
6. **Out of scope for the pattern, but open for reconstruction:** ECHO-style per-scan VarNet adaptation under real
   distribution shift (SPACE data), where zero-shot methods report their gains.

## Update: held-out per-volume parametric oracle, next step 1 (2026-10-19)
**Question.** Does a low-dimensional per-volume change to the mask pay on slices it was not fitted to? This is the
kind of pattern an ACS -> parameters predictor would output. The decision rule was set before the run: build such a
predictor only if `family - family_ctrl` is at least 0.1 dB with its 95% interval above 0.

**Method.**
- **Code and outputs.**
  - Code: `scripts/diagnostics/subject_coadapt_ceiling_r8.py` (`DensityFamily`).
  - Launchers: `analysis/subject_coadapt/run_param_family.sh` and `run_param_family_helper.sh`.
  - Per-volume results: `analysis/subject_coadapt/{brain,knee}_block_param/`.
  - Tables: `analysis/subject_coadapt/summary_param*.md`.
- **Data and split.** Same frozen LOUPE VarNet, block split, gap, slice counts and seeded draws as above, on all 23
  brain and 10 knee validation volumes. On overlapping volumes the base PSNR matched the earlier block-split runs
  exactly.
- **Two families.** Both are a density ratio on LOUPE's mask, p = p_LOUPE x exp(delta(theta)), so they equal LOUPE
  exactly at theta = 0 (asserted).
  - `param`, 4 numbers: radial scale, anisotropy along any axis (2 numbers), radial density tilt. Scale and
    anisotropy warp LOUPE's density envelope (an 8 px blur), so LOUPE's pixel structure is kept.
  - `grid`: an 8 x 8 bicubic offset.
- **Fits.** The free mask arm's recipe: 150 Adam steps, 4 slices x 2 draws per step, lr 0.02.
  - *on F* (scored on H);
  - `_in`: on F and H;
  - `_ctrl`: on as many other-subject training slices;
  - `_h`: on H alone, parameters only.

  The free maps `mask` and `mask_in` (lr 0.05) are anchors.
- **Reliability.** Across volumes, the correlation of the parameters fitted on F with those fitted on H. The
  other-subject fit against H is a null. For the grid, the within-volume correlation of the two centred grids is
  compared with the between-volume one.

| Contrast (dB, slice-weighted [95% CI over volumes]) | Brain (23 volumes, 129 slices) | Knee (10 volumes, 80 slices) |
|---|---|---|
| `param - param_ctrl` (decision contrast) | +0.008 [-0.002, +0.016] | -0.024 [-0.059, +0.007] |
| `grid - grid_ctrl` (decision contrast) | -0.042 [-0.131, +0.019] | -0.063 [-0.141, +0.013] |
| `param` - base (held out) | +0.003 [-0.004, +0.010] | -0.040 [-0.066, -0.013] |
| `grid` - base (held out) | -0.048 [-0.135, +0.011] | -0.084 [-0.155, -0.020] |
| `param_in` - base (in-sample) | +0.008 [-0.001, +0.016] | +0.005 [-0.013, +0.028] |
| `grid_in` - base (in-sample) | -0.002 [-0.046, +0.032] | +0.016 [-0.009, +0.044] |
| `param_ctrl` / `grid_ctrl` - base | -0.004 / -0.007 | -0.016 / -0.021 |
| `mask_in` - base (free per-volume map, in-sample) | +0.243 [+0.208, +0.332] | +0.170 [+0.103, +0.228] |
| `mask` - base (free map, held out) | -0.141 [-0.196, -0.090] | -0.116 [-0.171, -0.063] |

**Robustness.**
- Without the low-PSNR brain volume `AXT2_201_2010571`, brain `param - param_ctrl` is +0.005 [-0.004, +0.013] and
  `grid - grid_ctrl` is -0.049 [-0.139, +0.014].
- Also without `AXT2_200_6002375`, they are +0.006 [-0.003, +0.015] and -0.006 [-0.032, +0.019]. On that volume the
  grid fit lost 0.86 dB held out and 0.41 dB in-sample: it moved the low-frequency nodes by factors of e^+-1.

| Parameter | Brain: corr(F fit, H fit) | Brain: corr(other-subject fit, H fit) | Knee: corr(F fit, H fit) | Knee: corr(other-subject fit, H fit) | Spread of the F fit across volumes, brain / knee |
|---|---|---|---|---|---|
| radial scale | +0.60 | +0.04 | +0.43 | -0.41 | 0.071 / 0.038 |
| anisotropy a | +0.78 | -0.03 | +0.58 | -0.43 | 0.071 / 0.060 |
| anisotropy b | +0.72 | -0.48 | +0.66 | +0.72 | 0.034 / 0.092 |
| radial tilt | +0.68 | +0.09 | +0.52 | -0.26 | 0.098 / 0.091 |
| 8 x 8 grid, centred: within / between volumes | +0.45 / -0.01 | | +0.56 / -0.06 | | |

- **No-go by the rule set before the run.**
  - Every decision contrast is at most +0.008 dB, and no interval reaches +0.02.
  - Held out, the fitted families tie or lose against LOUPE (knee `param` -0.040, `grid` -0.084 dB).
- **A low-dimensional density change cannot gain even in-sample.**
  - `param_in` gains +0.005-0.008 dB, about 3% of the free per-volume map's in-sample +0.17 to +0.24 dB.
  - So the per-volume oracle's in-sample gain is high-dimensional structure fitted to those slices, not a change of
    density shape. This matches the per-slice picture in `reports/ADAPTIVITY_MECHANISMS_R8.md` (2026-10-19
    sections).
- **Each volume has a reproducible preferred density shape, but PSNR is flat along it.**
  - A volume's two halves agree on its parameters (Pearson 0.43-0.78; grid 0.45-0.56 against about 0 between
    volumes). The other-subject fits do not. The knee anisotropy-b null of +0.72 is one of eight null correlations,
    each over only 10 or 23 volumes.
  - **Robustness.** Scale and the two anisotropy terms hold up:
    - Spearman 0.41-0.56;
    - the smallest leave-one-volume-out Pearson is 0.30 or more;
    - brain Pearson 0.46-0.72 without the two flagged volumes.

    The radial tilt is fragile in brain: Spearman 0.27, and 0.12 without those volumes. The grid's two-halves
    correlation is positive in 83% (brain) and 100% (knee) of volumes, median 0.56-0.60.
  - Even fitted with the scored slices, the shape is worth under 0.01 dB. Whether the ACS carries it is untested and
    moot.
- **Fitting noise outweighs the signal.** Fits on 16 slices with 2 draws per step lose PSNR held out: the free map
  0.12-0.14 dB, the families up to 0.08 dB, and even the other-subject fits 0.004-0.021 dB.

**Conclusion.**
- Next step 1 is closed.
- Together with the per-slice result (`reports/ADAPTIVITY_MECHANISMS_R8.md`), no ACS -> pattern map is worth
  building on this data, per slice or per volume.
- What is left from the list above is anatomy or protocol routing from the header (step 2: about +-0.03 dB here, a
  sign flip under CG-SENSE) and budget adaptation (step 3).

**Caveats.**
- **One frozen VarNet,** the one co-adapted to the shared LOUPE mask. A VarNet fine-tuned per family was not tested.
  Earlier, VarNet and mask gains added with little amplification (`recon_on_mask_in`).
- **Fit length.**
  - The `param` fits settle by about 50 steps.
  - The grid's outer nodes still drift at 150 steps (Adam on tiny gradients), so its in-sample number might rise
    with longer fits.
  - Longer fits made the free map worse held out.
- **Parameter estimates** come from half a volume's ground truth, not from an ACS. That is the best case for a
  predictor.
- **Two brain volumes were computed twice.** A reverse-order helper shard overlapped the forward shard, and the
  saved result is the later run: same protocol, different random draws.
- **fastMRI 2D slice stacks** stand in for 3D SPACE volumes.
