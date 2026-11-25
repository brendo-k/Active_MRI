# Does an RSS loss make LOUPE learn partial Fourier, and what is the one-sided half worth? (2026-11-24)

| | |
|---|---|
| Code | results produced at `2a4d188` (loss_domain__loupe_loss_partial_fourier: phase-realness check of the combined target); branch `exp/loss_domain__loupe_loss_partial_fourier` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 8 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 58 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |

- **Status:** parked (2026-11-24). The owner agreed that the sensitivity maps absorb the object's phase, so this
  complex loss did not test whether a phase-aware loss removes partial Fourier; later losses act on the uncombined coil
  images, whose phase the maps cannot absorb. The value-of-the-half and mechanism results stand.
- **Branch:** `exp/loss_domain__loupe_loss_partial_fourier` @ `fc58fef5c877ff3d0c874ddbba404319cb547df7`. Return to the code with:
  `git worktree add ../SeqMRI-worktrees/exp-loss_domain__loupe_loss_partial_fourier exp/loss_domain__loupe_loss_partial_fourier`
- **Run state:** all 8 runs trained at run state 2 (`24db7c7`); every result
  file was regenerated at `227b0c4` (analysis script, `run.sh` and the extent check); `2a4d188` adds only the
  realness check and its output.
- **Frozen rules:** `proposal.md` (2026-11-19) and `addendum_2026-11-20_probe.md` (relative placebo bar), both set before
  any arm trained.

**Proposal vs outcome** (both seeds required for every rule)

| Readout | Frozen rule | Seed 42 | Seed 43 | Outcome |
|---|---|---|---|---|
| Guards | pooled SSIM ≥ 0.80; sym within 0.03 SSIM of free; L_cplx within 0.03 of L_rss; realised fraction 0.125 ± 0.002 | all pass | all pass | **met** |
| Layout (primary) | L_rss ≤ 0.15 and L_cplx ≥ 0.30: loss decides. L_cplx ≤ 0.15: it does not | L_rss 0.046, L_cplx 0.073 | L_rss 0.041, L_cplx 0.102 | **the loss does not decide the layout** |
| Value of the half under RSS | L_rss − L_rss_sym ≥ +0.10 dB, CI above 0 | +0.355 dB [+0.098, +0.562] | +0.316 dB [+0.067, +0.519] | **met**: partial Fourier is worth ≥ 0.1 dB |
| Value under the complex loss | reported; expected to include 0 | +0.355 dB [+0.164, +0.530] | +0.428 dB [+0.259, +0.582] | **not as expected**: CI above 0 |
| Mechanism (L_rss) | −k antilinear fraction ≥ 0.90, response ≥ 0.10 of self, ≥ 0.30 above each placebo | 0.947, 0.838, +0.483 / +0.524 | 0.943, 0.819, +0.446 / +0.428 | **met** |
| Complex target realness (added after the results, not frozen) | none | L_cplx target 95% real | L_cplx target 85% real | **the complex arm did not test the hypothesis** |

Layout numbers are the right-of-DC share (fraction of map mass, 0–1; 0.5 = symmetric). PSNR differences are paired over
the same 307 dev slices, 95% CI from resampling whole volumes.

**Recommendation:** park. The value-of-the-half and mechanism rows hold, but the layout row cannot be read as the
hypothesis intended. The complex target is formed through the model's own sensitivity maps, which are coil image ÷ RSS
and so absorb the object's phase: the combined target is 85–99% real, and a real image's k-space is conjugate-symmetric.
This complex loss is therefore close to a magnitude loss and cannot charge for partial-Fourier phase errors. The
follow-up uses a loss on the uncombined coil images, whose phase the maps cannot absorb.

## Question

Learned LOUPE maps on FLAIR are symmetric within about 15 columns of the k-space centre (DC) and one-sided beyond: the
partial-Fourier layout. Hypothesis: the RSS-magnitude loss causes this, because it does not charge for the phase errors
a one-sided acquisition creates, while a complex-image loss does. The experiment also measures what the one-sided half is
worth, by forcing the map to be point-symmetric (equal probability at k and −k), which rules partial Fourier out.
Columns are the phase-encode axis; DC is column 160 of the 320-column grid.

## Runs compared

FLAIR tier F (339 training volumes), R = 8, LOUPE + 6 × 18 E2E VarNet, 10 × 10 calibration centre, 100 epochs, lr 1e-3
with ×0.1 at epoch 90, batch 16, Muon + LayerNorm (main's default), trained on CoreHPC (jobs 3161844–51, about 4.2 h
each). Scored at `last_model.pt` (epoch 100) on the FLAIR dev half (20 volumes, 307 slices) with
`eval_multistep_adaptivity.py score`, draw seeds as in each score file. Checkpoints:
CoreHPC `~/scratch/loss_domain__loupe_loss_partial_fourier/runs/<run>`, lab copy
`/export/scratch/bkadota/loss_domain__loupe_loss_partial_fourier_sync/runs/<run>`.

| Run | Loss | LOUPE map | PSNR (dB) | SSIM | NMSE | Realised fraction |
|---|---|---|---|---|---|---|
| Lrss_1_s42_tF | RSS L1 (control) | free | 33.371 | 0.8907 | 0.00458 | 0.1250 |
| Lrss_1_s43_tF | RSS L1 (control) | free | 33.352 | 0.8908 | 0.00459 | 0.1250 |
| Lcplx_1_s42_tF | complex L1 | free | 31.365 | 0.8811 | 0.00901 | 0.1250 |
| Lcplx_1_s43_tF | complex L1 | free | 31.435 | 0.8813 | 0.00893 | 0.1250 |
| Lrsssym_1_s42_tF | RSS L1 | point-symmetric | 33.016 | 0.8864 | 0.00478 | 0.1250 |
| Lrsssym_1_s43_tF | RSS L1 | point-symmetric | 33.036 | 0.8868 | 0.00477 | 0.1250 |
| Lcplxsym_1_s42_tF | complex L1 | point-symmetric | 31.009 | 0.8761 | 0.00927 | 0.1250 |
| Lcplxsym_1_s43_tF | complex L1 | point-symmetric | 31.007 | 0.8755 | 0.00941 | 0.1251 |

- **Complex L1:** output Σ_c conj(S_c)·IFFT(k̂_c) against target Σ_c conj(S_c)·IFFT(y_c), both through the model's own
  sensitivity maps S (detached for the target), same loss region as RSS.
- **Point-symmetric map:** logit(i, j) = ½ [w(i, j) + w((320 − i) mod 320, (320 − j) mod 320)]. Points are still drawn
  independently; k and −k only share a probability.
- All metrics are against the RSS target. PSNR is compared only within a loss: the complex arms' outputs have no RSS
  noise floor, so their RSS-target PSNR is biased low (about −2 dB here).

## Methods

1. **Guards** from the score files (`reports/loss_domain__loupe_loss_partial_fourier/analysis/summarise.py tables`).
2. **Layout:** `scripts/diagnostics/loupe_map_asymmetry.py` on each run's learned map p(k) = sigmoid(slope · logits),
   before the budget rescaling. Right-of-DC share = the map's mass right of DC divided by its mass on both sides, in
   columns 32–98 from DC (`PRIMARY_BAND = (32, 99)`) and rows 33–287 (`|row − 160| ≤ 127`), outside the 10 × 10 block.
   This is the proposal's window. `kx_asymmetry_test.py` uses a different window and is not used.
3. **Value of the half:** `scripts/eval/eval_multistep_adaptivity.py contrast`, per-slice PSNR of the free arm minus the symmetric
   arm of the same loss and seed, pooled over the 307 slices; 95% CI from 2,000 resamples of whole volumes.
4. **Mechanism:** `scripts/diagnostics/conjugate_symmetry_probe.py` (the port named in the addendum; the proposal's
   `probe_conjugate_symmetry.py` is its October predecessor). On 24 evenly spaced dev slices × 6 sampled locations k
   (radius 20–60 px, n = 144 per point), the run's final mask is drawn once (seed 1234) and held fixed. k is perturbed by
   d = 0.05 · y(k) in every coil, four times (±d, ±i·d). At the unsampled −k and at two placebo points (k mirrored in
   rows only, and in columns only), the change in the reconstruction's k-space is split into a part linear in d and a
   part antilinear in d (∝ conj(d)). Antilinear fraction = |anti|² / (|lin|² + |anti|²), median over perturbations;
   response = |change| relative to the change at k itself. A conjugate mechanism answers at −k antilinearly; a response
   with no conjugate relation gives a fraction near 0.5. Lab server, one RTX A6000 per arm pair, minutes per arm.
5. **Acquired extent** (`summarise.py extent`): for every dev slice and every local tier-S training slice, the share of
   the layout window on each side of DC where any coil's k-space is non-zero.
6. **Complex target realness** (`reports/loss_domain__loupe_loss_partial_fourier/analysis/phase_realness.py`, added after the frozen results): on 24 evenly spaced dev
   slices, for L_rss and L_cplx in both seeds, the target t = Σ_c conj(S_c)·IFFT(y_c) through the run's own maps.
   Real share = energy of Re(t·e^{−iφ}) over energy of t inside the metric mask, φ the best single global phase.
   Conjugate-symmetric share = energy of the part of FFT(t · metric mask) that equals the conjugate of its mirror, over
   the total. Both are fractions, 0–1; a real image gives 1 for both. Repeated with coil 0's map phase multiplied back
   into t, which stands in for a map set that leaves the object's phase in the image.
7. **Figures** (`summarise.py curves maps recons`): training curves from the slurm logs; learned maps; the median-PSNR dev
   slice of Lrss_1_s42_tF with each seed-42 run's mask, reconstruction and error on one scale.

## Results

### Layout: every free map is one-sided, whichever the loss

| Run | 6–15 cols | 16–31 cols | 32–63 cols | 64–98 cols | **32–98 (primary)** | Conjugate overlap |
|---|---|---|---|---|---|---|
| Lrss_1_s42_tF | 0.343 | 0.266 | 0.083 | 0.000 | **0.046** | 0.247 |
| Lrss_1_s43_tF | 0.337 | 0.250 | 0.074 | 0.000 | **0.041** | 0.248 |
| Lcplx_1_s42_tF | 0.383 | 0.312 | 0.128 | 0.017 | **0.073** | 0.281 |
| Lcplx_1_s43_tF | 0.401 | 0.334 | 0.169 | 0.031 | **0.102** | 0.332 |
| sym arms (all 4) | 0.500 | 0.500 | 0.500 | 0.500 | **0.500** | 1.000 |

Right-of-DC share per band of distance from DC (fraction, 0–1). Conjugate overlap = Σ p(k) p(−k) / Σ p(k)²
(1 = point-symmetric, 0 = complementary halves). The complex loss moves the map slightly towards symmetry
(+0.03 to +0.06 in the primary band) but stays far below the 0.30 bar, in both seeds. `analysis/layout.json`.

### Value of the half: forbidding partial Fourier costs about 0.3–0.4 dB under either loss

| Contrast | Seed 42 (dB) | Seed 43 (dB) |
|---|---|---|
| Lrss − Lrsssym | +0.355 [+0.098, +0.562] | +0.316 [+0.067, +0.519] |
| Lcplx − Lcplxsym | +0.355 [+0.164, +0.530] | +0.428 [+0.259, +0.582] |

Paired pooled PSNR, 307 slices, 20 volumes. Against the 0.3–0.5 dB seed spread of joint LOUPE + VarNet training, these
are paired within seed and replicate across both seeds. SSIM moves the same way (+0.004 to +0.006), with CIs that include 0 in three of four contrasts. `contrasts.json`.

### Mechanism: RSS-trained VarNets fill −k by conjugate symmetry; complex-trained ones only partly

| Run | −k antilinear fraction | −k response (of self) | −k minus row placebo | −k minus column placebo |
|---|---|---|---|---|
| Lrss_1_s42_tF | 0.947 | 0.838 | +0.483 | +0.524 |
| Lrss_1_s43_tF | 0.943 | 0.819 | +0.446 | +0.428 |
| Lrsssym_1_s42_tF | 0.953 | 0.846 | +0.497 | +0.449 |
| Lrsssym_1_s43_tF | 0.880 | 0.759 | +0.394 | +0.378 |
| Lcplx_1_s42_tF | 0.717 | 0.289 | +0.229 | +0.210 |
| Lcplx_1_s43_tF | 0.596 | 0.356 | +0.118 | +0.082 |
| Lcplxsym_1_s42_tF | 0.688 | 0.320 | +0.230 | +0.282 |
| Lcplxsym_1_s43_tF | 0.631 | 0.414 | +0.058 | +0.131 |
| L_1_s42_fscr1_tF (fscr1: one-step LOUPE) | 0.927 | 0.824 | +0.453 | +0.437 |
| F_4_s42_fscr1_tF (fscr1: 4-step adaptive SeqMRI sampler) | 0.972 | 0.759 | +0.501 | +0.453 |
| Fb_4_s42_fscr1_tF (fscr1: F's data-blind twin) | 0.985 | 0.923 | +0.422 | +0.438 |

Fractions 0–1, medians over n = 144 perturbations. Placebo fractions are 0.41–0.57 for every run. The RSS arms clear
every bar; the complex arms answer at −k with a weaker, less conjugate response, below the 0.30 margin in both seeds.
`analysis/probe_*.json`, `probe_fscr1.json` (run state 1).

### Complex target: the model's maps make it nearly real

| Run | Real share, own maps | Conjugate-symmetric share, own maps | Real share, coil-0 phase back | Conjugate-symmetric share, coil-0 phase back |
|---|---|---|---|---|
| Lrss_1_s42_tF | 0.983 [0.977, 0.989] | 0.983 | 0.751 [0.531, 0.975] | 0.485 [0.168, 0.874] |
| Lrss_1_s43_tF | 0.987 [0.982, 0.993] | 0.986 | 0.762 [0.534, 0.976] | 0.484 [0.154, 0.894] |
| Lcplx_1_s42_tF | 0.952 [0.933, 0.964] | 0.951 | 0.762 [0.557, 0.951] | 0.474 [0.218, 0.891] |
| Lcplx_1_s43_tF | 0.848 [0.784, 0.886] | 0.848 | 0.753 [0.545, 0.941] | 0.478 [0.196, 0.837] |

Fractions 0–1, median [min, max] over 24 dev slices. Through the models' own maps, 85–99% of the target's k-space energy
is conjugate-symmetric, so the half that partial Fourier skips is almost fully predictable from the half it samples,
under either loss. With an object phase put back, about 48% is. The less real target (Lcplx_1_s43, 0.85) goes with the
most right-of-DC mass (0.102); one run, not a test. `analysis/phase_realness.json`.

### Training

Every run trained the full 100 epochs without divergence (`figures/curves.png`). The curves show the trainer's
validation PSNR (about 34.6 dB for L_rss), which takes its data range from the last batch; the tables use the scorer's
pooled PSNR (33.37 dB for the same model), which is the number reported. The RSS arms are flat by epoch 90
(within 0.01 dB of their final value, except Lrsssym_1_s42 +0.26 dB after an epoch-90 dip); the complex arms' trainer validation PSNR plateaus by about epoch 10 and fluctuates by about ±0.2 dB.
The complex arms' training loss differs by about 0.003 between seeds (free and symmetric alike), the RSS arms' by
< 0.0005.

## Conclusions

- **The RSS loss does not decide the partial-Fourier layout.** With a complex-image loss the learned map is still
  one-sided (right-of-DC share 0.073 and 0.102 against 0.046 and 0.041 under RSS; the bar was 0.30). This is the frozen
  rule's outcome.
- **Forbidding the one-sided layout costs about 0.3–0.4 dB at R = 8 under RSS:** L_rss − L_rss_sym = +0.355 and
  +0.316 dB, CIs above 0 in both seeds. The October estimate (−0.08 dB for a kx-symmetrised mask,
  `reports/PARTIAL_FOURIER_REPORT.md`) is not comparable: R = 4 brain, older data and metric.
- **RSS-trained VarNets fill the unsampled half by conjugate symmetry**, as the mechanism row requires (−k antilinear
  0.94–0.95, 0.43–0.52 above the placebos), and so do fscr1's L, F and Fb.
- **This complex arm did not test the hypothesis.** Its target goes through the model's own sensitivity maps, which
  absorb the object's phase, so the target is 85–95% real and its k-space 85–95% conjugate-symmetric. A loss on a
  near-real image charges little for skipping one half, so the one-sided map under this loss says nothing about
  whether a phase-aware loss would keep it. Measured after the results; the frozen rules did not anticipate it.
- **The complex arms' symmetric constraint costs as much** (+0.36, +0.43 dB) while their VarNets use conjugate filling
  less (−k antilinear 0.60–0.72). Hypothesis, untested: tying k to −k also costs through other routes (the symmetric map
  spreads its budget over a box twice as wide, `figures/learned_maps.png`), so 0.3–0.4 dB is an upper bound on what
  partial Fourier itself is worth. (Not a gradient effect: the mask logits are on Adam, whose step does not scale with
  gradient size.)
- **Nor does the raw data set the layout:** inside the window, acquired k-space covers 96.2% (left) and 95.9% (right)
  of the dev slices' window on average, and the same 232 of 307 dev slices are fully acquired on both sides (training
  tier S: 98.4% and 98.2%, 1,059 of 1,211 slices). `analysis/extent.json`.
- Hypothesis, untested: a loss that keeps the object's phase (on the uncombined coil images, or through fixed
  phase-referenced maps) removes the one-sided layout. Earlier complex-loss runs with a different map set gave two-sided
  maps, which fits this.

## Caveats

- **Cross-loss PSNR is not comparable.** The complex arms score about 2 dB lower against the RSS target because their
  output has no RSS noise floor (dark background in `figures/recons.png`), and their error concentrates at the skull
  rim and ventricles. Only within-loss contrasts are read.
- **The complex target is defined through each model's own detached maps**, so its phase and weighting are
  model-specific: 95% real in seed 42, 85% in seed 43. That may explain the seed-dependent complex training loss and
  the noisy validation curve.
- **The realness check was added after the results** and uses coil 0's phase as a stand-in for phase-keeping maps,
  not a real map set such as ESPIRiT.
- **The layout window** (columns 32–98, rows 33–287) is not fully acquired on every slice: 75 of 307 dev slices miss
  part of it (at least 79% acquired on each side), symmetrically. Training slices were checked on local tier S
  (1,211 slices), not the full tier F.
- **Two complex runs have 2 slices below 20 dB** (Lcplx_1_s42 and Lcplxsym_1_s43; `file_brain_AXFLAIR_202_6000450`
  slices 12 and 14). The same slices are the hardest for every arm (about 26 dB under RSS).
- **Median slice of the recon figure** favours the symmetric RSS map (34.51 against 33.45 dB); one slice, opposite to the
  pooled result.
- **Probe:** 24 slices × 6 locations; fscr1's probe ran at run state 1 with the same script and settings. The
  perturbation (5% of y(k)) is small but still an input the network never saw; a positive result is direct evidence,
  a negative one would be weaker.
- **Recon figure** loads training tier S only to build data loaders (tier F's training files are not on the lab
  server); the dev set is the same.
- wandb runs were logged offline and synced on 2026-11-24 (SeqMRI project, ids in each run's `wandb/` folder).

## Figures

- `figures/curves.png`: trainer validation PSNR and training loss per epoch, all 8 runs.
- `figures/learned_maps.png`: every run's learned map p(k), with the layout window dotted.
- `figures/recons.png`: median-PSNR dev slice, seed-42 masks, reconstructions and errors on one scale.
- `figures/scorer/<run>_q50.png`: the scorer's own figure per run (median-PSNR slice: input, map, loss gradient, new
  samples, mask, zero-filled, reconstruction, error).

## Next steps

- Separate partial Fourier from the cost of tying logits: a symmetric arm whose logits are free but whose draws are
  mirrored (sample k ⇒ sample −k at half the budget), or an RSS-free map with the right half masked out (explicit partial
  Fourier) against the free map.
- Test the hypothesis with a phase-aware loss: L1 on the uncombined coil images (the coil-image arms of branch
  `exp/multistep_adaptivity__seqmri_loss_domain_r8`), whose phase the maps cannot absorb, and check whether
  the right-of-DC share rises above 0.30. Report the target's real share for every loss arm.
- The D4-augmentation proposal (multistep_adaptivity line) rotates this wedge with the data; its upright-vs-augmented
  readout will show how much the one-sided prior is worth to the multi-step samplers.

## Reproduce

`bash reports/loss_domain__loupe_loss_partial_fourier/run.sh` regenerates the contrasts, tables and curves from the
score files (no GPU). With `WITH_CHECKPOINTS=1` (and `RUNS_DIR`, `DATA_ROOT` if not on the lab server) it also reruns the
layout, the probe, the extent check, the realness check and the map and recon figures (one GPU, about 20 min). Check:
`python scripts/report_manifest.py check reports/loss_domain__loupe_loss_partial_fourier`.
