# Can the reconstructor fill in the missing half of k-space? (2026-10-04)

## Question
Learned LOUPE brain masks are strongly asymmetric along kx: almost no probability on the right ~40% of columns.
Is the network exploiting conjugate (Hermitian) symmetry, i.e. estimating low-order phase and skipping -k?
Baseline: wandb `brenden-kadota-university-of-california-sf/SeqMRI/db7fsabe`.

## Setup common to all runs
256x256, VarNet, brain-only training (`brain_knee_balanced`, validation on brain + knee), R=4 (sparsity 0.25),
lr 1e-3, weight decay 1e-4, bs 4. Baseline: 12 cascades, CAIPI 2x2 lattice, 1 step, 50 epochs, no LR schedule.
Final baseline brain val: NMSE 0.01354, PSNR 35.82, SSIM 0.9255.

## Evidence already in the repo (not rerun this session)
Source: `runs/loupe_anatomy/analysis`, 12-cascade zero-step LOUPE 2D, 3 seeds per anatomy.
- Brain seed-mean mask is ~zero on the right of kx and tapers triangularly; knee mask is near-symmetric.
- Flipping the mask in kx costs ~0.22 dB PSNR (L1 +0.0018); ky flip costs ~0.002 dB; point flip equals kx flip.
  A kx-symmetrised mask is ~0.08 dB worse than the learned one.
- Seeds agree (smoothed r 0.97-0.99). Consistent with a partial-Fourier/homodyne-like use of conjugate symmetry along kx.
- `probe_conjugate_symmetry_out/summary.json`: coil-combined brain image is nearly real (realness 0.01-0.02 with smooth
  sense maps), which is what makes Hermitian symmetry usable.

## Code change
New option `sampler.caipi_keep_fraction` (default 1.0): CAIPI lattice points in the rightmost `1 - f` of columns are dropped.
f=0.75 keeps columns 0-191, drops 192-255; net acceleration 5.33x over the full grid. Verified that columns 0-191 equal the
full lattice. Files: `samplers/caipi.py`, `samplers/sampler.py`, `models/sequential_sampling.py`, `config.py`,
`training/trainer.py` (run name tag `pf0.75`), `scripts/train_sequential.py`, `configs/config.yaml`.
Launcher: `scripts/train_varnet_caipi_brain_only_pf6of8.sh`.

## Experiments launched this session
| # | Run | Setup | Status | Result so far |
|---|-----|-------|--------|---------------|
| 1 | `checkpoints/varnet_caipi2d_r4_brain_only_pf6of8` | 12 cascades, CAIPI, 6/8, 50 ep, GPU 3 | running (44/50 epochs at last check) | brain val NMSE ~0.0146, PSNR 35.33, SSIM ~0.922 (baseline 0.0135 / 35.82 / 0.9255) |
| 2 | `..._casc1_full`, `..._casc1_pf6of8` | 1 cascade, CAIPI full vs 6/8, GPU 2 | **killed by me after a few minutes** | none; partial dirs remain in `checkpoints/` |
| 3 | `runs/loupe_casc1/brain/seed_0` | 1 cascade, LOUPE 2D, zero-step, `datasets=brain`, 30 ep, GPU 2 | running (5/30 epochs at last check) | brain val NMSE 0.0117 at epoch 5; not comparable with other rows (different val set, n=587 vs 344) |

## Interpretation so far (tentative)
- Dropping 25% of the lattice costs only ~8% in NMSE (0.0146 vs 0.0135), so the 12-cascade network recovers most of the
  missing right-hand k-space. This is consistent with, but does not prove, conjugate-symmetry exploitation; it could be
  interpolation. The antilinearity test in the probe script separates the two.
- Why literature rarely shows voids (hypothesis, not checked against papers): fixed/symmetric masks leave no freedom to
  create them; this pipeline's near-real coil-combined phase; and a high-capacity 12-cascade network.

## Next steps
1. Let runs 1 and 3 finish; compare run 1 to `db7fsabe` using the same epoch and best-checkpoint metric.
2. Run `scripts/probe_conjugate_symmetry.py` on the baseline and 6/8 checkpoints. Add a test: error inside the dropped
   band (cols 192-255) vs elsewhere, and antilinearity (response at -k to perturbing k).
3. Compare to CG-SENSE and homodyne/POCS partial-Fourier on the same 6/8 mask, to put the network's NMSE in context.
4. Capacity test, redone cleanly: 1-cascade CAIPI full vs 6/8 (killed in run 2). The 6/8 gap vs the full-lattice control is
   the quantity of interest, compared with the 12-cascade gap.
5. For the LOUPE 1-cascade run: use the same `datasets=brain` split as `runs/loupe_anatomy` for a fair comparison, check
   whether the right-side void persists, add seeds 1-2, then run `scripts/analyze_loupe_anatomy.py` and `eval_flipped_mask.py`.
6. Optional: a 12-cascade LOUPE run with the 6/8 constraint, to see whether the learned mask moves toward a more uneven
   kx profile when part of k-space is unavailable.

## Caveats
- Run 3 and the earlier LOUPE analysis use `datasets=brain`; runs 1 and the baseline use `brain_knee_balanced`.
- Single seed for every run this session. Differences of a few percent in NMSE are within what seed noise could produce
  (not measured).
- "Right" = columns 192-255, confirmed by the user. The `num_cascades=1` setting was not verified in the training log.

## Results addendum (runs 1 and 3 finished, probes run)
Final brain validation (last epoch):
| run | brain NMSE | PSNR | SSIM | note |
|-----|-----------|------|------|------|
| baseline `db7fsabe` (12 casc, CAIPI full) | 0.01354 | 35.82 | 0.9255 | |
| 12 casc, CAIPI 6/8 | 0.01388 | 35.68 | 0.9238 | +2.5% NMSE with 25% fewer samples (the ~8% gap at epoch 17 closed) |
| 12 casc, LOUPE (`runs/loupe_anatomy/brain`, 3 seeds) | 0.0097-0.0098 | 34.7 | 0.938 | `datasets=brain`, n=587 |
| 1 casc, LOUPE (`runs/loupe_casc1`, seed 0) | 0.01064 | 34.08 | 0.934 | same split; +9% NMSE vs 12 casc |

Probes: `probe_out_caipi_pf6of8_casc12/`, `probe_out_loupe_casc1_brain_s0/` (single seed, 21 slices, 4 volumes).
- 1-casc LOUPE mask is still strongly one-sided in kx (right ~half empty except a thin ky~0 stripe); capacity does not remove it.
- Antilinearity (test 6, 1-casc LOUPE): a perturbation at k gives a response at -k that is 99.98% conjugate (antilinear) at
  19% of the self-response amplitude; placebo locations respond at <1% amplitude. Strong evidence of a conjugate mechanism
  in a 1-cascade network (n=36 perturbations).
- Test 3 (k-space error by partner location, 1-casc LOUPE): Hermitian-partner error is not lower than the mirror_ky placebo
  (gap -0.18 to -0.24 at r>=64); confounded because the mask is itself asymmetric. Not supportive, not conclusive.
- Test 4: flipping the mask costs 1-casc VarNet ~4.7% NMSE vs ~0.9% for CG-SENSE, so the network is somewhat mask-specific.
- CAIPI 12-casc probe is mostly degenerate: the lattice is point-symmetric so there are no mirror_ky placebo samples
  (gap = -1.0 is an artifact), antilinearity returned no rows, and realness (rss_norm) is 0.92 vs 0.017 for LOUPE, which
  suggests the sense maps estimated from a CAIPI centre (no fully sampled ACS) are poor. Treat it as uninformative.

## Addendum: 1-cascade LOUPE without the sense mask (2026-10-04)
Same run as `runs/loupe_casc1/brain/seed_0` except `+sampler.sense_mask_method=null` (sense maps no longer zeroed outside the
signal region). Checkpoint `runs/loupe_casc1_nosensemask/brain/seed_0` (best epoch 26), probe `probe_out_loupe_casc1_nosensemask_brain_s0/`
(same 21 slices, single seed). Final-epoch brain val NMSE 0.00476 / PSNR 36.14 / SSIM 0.972 -- NOT comparable with the masked run
(0.01064 / 34.08 / 0.934), because the trainer's metric masking depends on the sense mask.

| probe | masked (`..._brain_s0`) | unmasked (`..._nosensemask_brain_s0`) |
|-------|------------------------|----------------------------------------|
| realness, rss_norm (Im^2/\|x\|^2) | 0.017 | 0.162 |
| mask asymmetry, r 64-96 / 96-128 | 0.87 / 0.96 | 0.81 / 0.93 |
| antilinear fraction at -k | 0.9998 | 0.9996 |
| response at -k / response at k | 0.19 | 0.32 |
| placebo response (ky / kx) | <1% | 1.8% / 2.4% (antilinear fraction 0.45 / 0.21, so not conjugate) |
| NMSE learned -> kx/point flip | +4.7% | +66% (0.0108 -> 0.0179) |
| NMSE learned -> Hermitian-paired mask | +18% | +118% (0.0108 -> 0.0236) |
| Test 3 gap (herm vs placebo), r 16-96 | -0.01 to -0.24 | -0.02 to +0.05 |

- The one-sided kx mask survives removing the sense mask, so it is not an artifact of masking.
- The conjugate response is as strong or stronger without the mask (32% of the self-response, 99.96% antilinear).
- The coil-combined image is much less real (0.16 vs 0.017), so a near-real image is not required for the network to learn a one-sided mask.
- This network depends far more on its own mask: flipping it costs +66% NMSE vs +4.7%. The mask and reconstructor are co-adapted.
- Test 3 is still not decisive: the gap is within about +/-0.06 of zero for r < 96, and the mask itself is asymmetric. The 128-182 row
  (gap 0.71) is an artifact of almost no sampled points at those radii (both/null 1.56).
- CG-SENSE is much worse here (NMSE 0.21 vs 0.07 with the mask), which is consistent with unmasked sense maps being poor outside the
  signal support. Treat the varnet/cg ratios as incomparable across the two runs.
- Single seed and 21 slices; the differences between the two runs are not seed-controlled.
