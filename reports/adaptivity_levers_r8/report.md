# Where can adaptive sampling gain at R=8? Encoding census, multi-step evidence, literature regimes and proposed experiments (2026-11-16)

| | |
|---|---|
| Code | results produced at `7b92c6b` (adaptivity_levers_r8: encoding census script; report manifests may declare no checkpoint runs); branch `report/adaptivity_levers_r8` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | none: this report loads no checkpoint |
| Results | 10 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |

**Branch:** `report/adaptivity_levers_r8` (worktree `../SeqMRI-worktrees/report-adaptivity_levers_r8`), branched from `main` at `ac56a1f`.

**Document type:** research synthesis and proposal, plus one CPU-only, training-free census. No GPU run and no checkpoint
is loaded. The census reports linear-SENSE noise amplification, not VarNet PSNR; the metric mask enters only as the
object support. The multi-step numbers are snapshots of results the branch `exp/multistep_adaptivity` produced
(tip `c718a3d`); they are trainer-metric PSNR under the same apodized mask.

## Question

The project wants a sampler plus reconstructor that adapts to the anatomy and the objects in the field of view. Measured
adaptive gains so far are 0.05-0.2 dB against 0.3-0.5 dB seed spread. Which adaptive levers can still give a larger
benefit, what do the papers that report adaptive gains have in common, and what should be run next?

Conventions: `f` = fraction of the 320 x 320 FOV inside the object support; `g` = SENSE g-factor; "noise dB" =
10 log10 of the object-mean g^2; R = 8 unless stated; a "twin" is the same model with the data-dependence removed.

## Runs compared

No checkpoint is loaded by this report.

| Source | What it is | Setting |
|---|---|---|
| Census (this report) | 120 validation volumes of local v1 (`configs/datasets/fastmri_local_v1/manifest.csv`, `split == val`): 40 T2, 40 FLAIR, 20 knee PD, 20 knee PDFS | read as the trainer reads them: readout oversampling cropped, 320 x 320 k-space, brain row flip |
| `exp/multistep_adaptivity` @ `c718a3d`, kscr1 | A_16, T_16 and a floor twin, seed 42, 12-cascade VarNet | knee PD + PDFS, R=8, 100 epochs, 16 steps, CoreHPC |
| `exp/multistep_adaptivity` @ `c718a3d`, kscr2 | T_1, T_16, A_16, S_16, C_16, seed 42, 6-cascade VarNet | same recipe; scored on the dev half of knee val (669 slices, 20 volumes) |

Health of the multi-step runs (from that branch's `kscr2/README.md`): pooled PSNR 34.67-35.02 dB, pooled SSIM
0.864-0.872 (knee-PDFS 0.80-0.81 per condition), realised sampling fraction 0.125, no slice below 20 dB.

## Methods

Everything in items 1-4 is `scripts/diagnostics/encoding_census_r8.py`; `summarize` writes `summary.json` and `summary.md`.

1. **Support census** (`support`, every 2nd slice). Per slice: `f`, extents, and for each of the 15 index-8 sublattices
   of the k-space grid the object-weighted number of aliased object pixels `E_Ω[n]`. A lattice of index R folds R image
   shifts onto each pixel; `n` counts how many of them land inside the object.
2. **g-factor census** (`gfactor --acceleration 8` and `4`, every 3rd slice). Sensitivity maps = Hann-windowed 24 x 24
   centre divided by its RSS. Per object pixel, `g^2 = [(S^H S)^-1]_ρρ [S^H S]_ρρ` over the folded object pixels of that
   lattice, capped at 100, no noise prewhitening. "Group-best" is the lattice with the lowest group mean; "own-best" is
   the lowest for that volume.
3. **Budget estimate** (`summarize`). Linear-model slope per volume `b_v = 3.01 + (noise dB at R=8 − noise dB at R=4)`
   dB per octave of samples, each at the volume's own-best lattice. Water-filling at a fixed cohort total gives
   `n_v ∝ b_v` and a mean gain `mean_v[b_v log2(b_v / b̄)]`.
4. **Region-specific check** (`roi`, every 4th slice, 13 non-degenerate lattices). Mean g^2 inside 32-pixel discs at
   the support centroid and offset by a quarter of the extent up, down, left and right.
5. **Multi-step evidence.** `multistep_kscr1_contrasts.txt`, `multistep_kscr2_contrasts.txt`,
   `multistep_kscr2_leak_gap.txt` and `multistep_kscr2_leak_swap.txt` are copied by `run.sh` from
   `reports/multistep_adaptivity/` at `c718a3d`. Arms: T_1 = one-step static LOUPE; T_16 = 16-step data-blind twin;
   A_16 = sampler reads the measured k-space magnitude; S_16 = also reads complex ACS virtual coils and an in-loop
   VarNet reconstruction (SeqMRI-style); C_16 = sampler reads the full k-space (clairvoyant). CIs resample volumes.
6. **Literature.** Eight papers named by the user, read for coils, mask type, reconstructor, steps, objective and
   whether a non-adaptive twin exists. Sources are at the end; SeqMRI, SUNO and DCM numbers come from
   `reports/LITERATURE_ADAPTIVE_SAMPLING_R8.md` and earlier notes on the DCM PDF.

## Results

### 1. Encoding difficulty differs strongly between scans; the best mask shape does not

| Group (volumes) | Coils | f mean (range) | 8·f range | Noise at R=4 | Noise at R=8 (volume range) | Own-best vs group-best lattice, mean (max) |
|---|---|---|---|---|---|---|
| T2, many coils (36) | 12-20 | 0.35 (0.29-0.43) | 2.4-3.4 | 0.4 dB | 3.5 dB (2.4-5.8) | 0.006 dB (0.09) |
| FLAIR, many coils (36) | 12-20 | 0.39 (0.25-0.58) | 2.0-4.7 | 0.6 dB | 4.2 dB (2.3-6.3) | 0.021 dB (0.15) |
| Knee PD (20) | 15 | 0.57 (0.50-0.72) | 4.0-5.8 | 1.0 dB | 5.9 dB (4.6-7.7) | 0.051 dB (0.29) |
| Knee PDFS (20) | 15 | 0.60 (0.51-0.74) | 4.1-5.9 | 1.3 dB | 6.3 dB (4.9-8.4) | 0.074 dB (0.36) |
| T2, 4 coils (4) | 4 | 0.33 (0.32-0.36) | — | 1.1 dB | 11.6 dB (10.2-13.1) | 0.052 dB (0.20) |
| FLAIR, 4 coils (4) | 4 | 0.40 (0.34-0.47) | — | 1.6 dB | 14.1 dB (10.1-17.5) | 0.180 dB (0.57) |

The same nominal R=8 costs 2.3 to 17.4 dB of noise depending on the scan, while choosing the lattice per patient
instead of per group is worth 0.006-0.07 dB with many coils (0.18 dB for 4-coil FLAIR, 4 volumes).

- **Support alone.** The best lattices sit at 1.16-1.19 x the bound `E_Ω[n] ≥ R·f` in every condition, and own-best
  minus group-best is at most 0.017 aliased pixels. Support does not pick a different lattice per patient.
- **Predictable from the prescan.** Noise dB ≈ a + 2.35·log2 f − 4.00·log2 coils, R^2 0.86 (120 volumes; coil count is
  confounded with scanner model).
- **Four-coil brains are a different regime.** 5.7% (T2) and 21.6% (FLAIR) of their object pixels fold more aliases
  than there are coils, so SENSE cannot resolve them; 67% of pixels have g > 2, against 4-10% for many-coil brains.
- **Within a volume** `f` varies a lot between slices (slice sd 0.13 brain, 0.24 knee).

### 2. Budget reallocation under the linear model

| Quantity | Value |
|---|---|
| Slope b, all volumes | 7.40 dB/octave, sd 2.18, CV 0.29 |
| Slope by group | many-coil brain 6.1-6.6; knee 7.8-8.0; 4-coil brain 13.5-15.3 |
| Slope fitted from log2 f and log2 coils | R^2 0.87 |
| Water-filling at a fixed cohort total | +0.39 dB mean, R_v from 3.3 to 11.8 |
| Second-order estimate `b̄·CV^2/(2 ln 2)` | +0.46 dB |
| Samples to hold every volume at the 90th-percentile volume's penalty | 0.80 x fixed R=8 |

The reallocation gain on the mean is second order in the slope spread and comes mostly from the eight 4-coil volumes;
the equal-quality saving is first order.

### 3. Region-specific fidelity

| Group (volumes) | Patient-specific gain, whole object | Patient-specific gain, region | Static region-specific lattice | Volumes with zero region gain |
|---|---|---|---|---|
| Brain, many coils (72) | 0.02 dB | 0.10 dB | 0.09 dB | 49% |
| Knee (40) | 0.08 dB | 0.16 dB | 0.09 dB | 63% |
| Brain, 4 coils (8) | 0.09 dB | 0.40 dB | 2.56 dB | 78% |

Scoring a region instead of the whole object raises the per-patient effect several-fold, but with many coils it stays
at 0.1-0.2 dB and about half the volumes gain nothing.

### 4. Multi-step sampling in this repo (`exp/multistep_adaptivity` @ `c718a3d`, knee, seed 42)

Pooled PSNR differences in dB with 95% CIs (kscr2, 6 cascades, unless marked):

| Contrast | Pooled | knee-PD | knee-PDFS | Reading |
|---|---|---|---|---|
| A − T_16 (kscr1, 12 cascades) | +0.077 [+0.058, +0.094] | +0.102 | +0.051 | adaptivity; floor twin +0.002 [−0.013, +0.016] |
| A − T_16 | +0.059 [+0.043, +0.074] | +0.064 | +0.054 | adaptivity proper |
| S − T_16 | +0.060 [+0.033, +0.085] | +0.013 | +0.108 | in-loop reconstruction + complex input |
| S − A | +0.001 [−0.021, +0.023] | −0.051 | +0.053 | the SeqMRI-style loop adds nothing pooled |
| T_16 − T_1 | +0.044 [+0.033, +0.056] | +0.057 | +0.030 | data-blind 16-step spacing |
| A − T_1 | +0.103 [+0.080, +0.123] | +0.122 | +0.084 | gap to one-step LOUPE |
| C − A, each through its own VarNet | +0.247 [+0.207, +0.286] | +0.318 | +0.174 | clairvoyant ceiling at face value |
| C − A, both through T_16's VarNet | −0.083 [−0.150, −0.033] | −0.058 | −0.108 | the ceiling does not survive a neutral decoder |
| Swap cost C − A, through T_16's VarNet | +0.184 [+0.078, +0.272] | +0.364 | +0.001 | C's choices are slice-matched, knee-PD only |

Sixteen steps of feedback are worth about +0.06 dB over the data-blind twin; 43% of the +0.103 dB gap to one-step
LOUPE is data-blind spacing; and the clairvoyant arm's +0.25 dB was its own VarNet reading the mask as a code.

- **Swap cost is not a benefit measure.** S raised its swap cost over A (+0.189 against +0.113) with no gain over the
  twin, because a VarNet co-adapted to an arm's own masks raises it too.
- **Where the data-dependence sits.** A moves 9% of a step's draws at step 0, 14-18% in steps 1-10 and 37% at step 14
  when the data is swapped; S moves 35% already at step 0, before any reconstruction.
- **Not data-limited** (branch note): S's training loss is no lower than A's and the val/train ratio is 1.065-1.072 in
  every arm.

### 5. What the cited papers' gains come from

| Paper | Coils, grid | Mask | Reconstructor | Steps | Objective | Non-adaptive twin | Headline |
|---|---|---|---|---|---|---|---|
| SeqMRI (Yin 2021) | 1 coil, 128² | 2D points, 1D lines | U-Net | 1-4 | SSIM | 1-step only | 8x: 91.07 vs LOUPE 90.60; 1-step +0.07 points over its twin |
| L2SR (Yang & Dong 2024) | 1-coil knee 128²; brain 256² | 1D columns | 16-channel U-Net on zero-filled magnitude | many | SSIM | no | knee 8x 0.6712 vs LOUPE 0.5876; its LOUPE is at or below random |
| SUNO (Gautam 2026) | multi-coil | 1D lines | VarNet, MoDL | 1 (lookup) | PSNR | partly | brain 8x +0.27 dB vs LOUPE; knee 8x −0.42 dB vs BASS |
| Region-specific RL (Xu & Öksüz 2025) | ACDC, fastMRI knee, 4x/6x | lines | not read | many | global SSIM + region metrics + k-space MSE | random only | abstract only |
| ASMR (Yen 2024) | 1 coil | 1D columns | none (classifier on k-space) | about 50 | pathology AUROC | no | knee +2-5 AUROC points vs LOUPE/EMRT; prostate loses to EMRT |
| Hong et al., ECCV 2024 | 16 coils, 320² | 2D and 1D | 5-cascade VarNet, one per mask (3 pairs) | 1 (select) | SSIM | static variable density | 2D 8x +0.0007, 1D 8x +0.004, CelebA 8x +0.033 SSIM |
| Wang, Noll & Fessler 2023 | multi-coil (ESPIRiT) | 1D and 2D | posterior sampling, roughness or score prior | 30-50 | PSNR | no learned static mask | 1D 10x 38.9 vs 35.2 dB (20 slices); 2D 12x 40.1 vs 38.9 dB Poisson-disc |
| Hong GQ et al. 2023 (DuDReTLU-net) | image-derived k-space, 128² cine | population LOUPE, one mask per frame | dual-domain transformers | 0 | MSE | not patient-adaptive | 5x 38.86 vs LOUPE 35.64 dB; 20x 33.88 vs 29.87 dB |

Positive results cluster on line masks, one coil, weak reconstructors, many steps, or task/region objectives.

- In the one paper with 2D masks, 16 coils and a VarNet (Hong ECCV), the gain over the best static mask is under
  0.001 SSIM; without coils the same method gains 0.033.
- Wang et al.: with the analytical (Gaussian roughness) prior the posterior covariance does not depend on the data, so
  those patterns adapt to the coil maps and the sampled set, not to image content (a reading of their model, not a
  claim in the paper). Their dynamic case designs the pattern on a pilot frame of the same object.
- DuDReTLU-net is a stronger reconstructor with population masks that are complementary across five correlated frames.
- DCM (Sun 2024): in the 2D, data-consistent regime its gap to LOUPE was 0.01-0.10 dB, but its instability premise
  matches earlier results here: trained maps keep the starting draw's layout (correlation 0.4-0.7), uniform-start seeds
  differ by 36% of the budget, the straight-through gradient is blind to radial density (+0.25 dB from density alone
  in CG-SENSE PDFS), and dead logits cannot revive.

## Conclusions

- **Mask shape at a fixed budget is a flat direction with many coils.** The census gives ≤0.07 dB per-patient noise
  gain; earlier reports moved 29-50% of the budget within 0.1 dB. Near an optimum the adaptive gain is
  `½ E_v[(m̄ − m_v*)ᵀ H_v (m̄ − m_v*)]`: second order in the spread of optima, times a small curvature.
- **Measurement feedback does not change that in the current regime.** Sixteen steps give +0.06 dB over the twin, an
  in-loop reconstruction adds +0.001, and the clairvoyant ceiling was a mask code. How much a better partial-data
  sampler could gain is unknown and well under +0.25 dB.
- **Encoding difficulty is the heterogeneous quantity.** It spans about 15 dB at R=8 and `f` and coil count explain
  86% of it, so budget (R per scan) is the lever with the most between-patient spread. Its mean-PSNR gain is second
  order; its tail and equal-quality gains are first order.
- **The regimes where the literature finds adaptivity differ from this one on four axes:** line masks, one coil, a weak
  reconstructor, and a task or region objective. The fifth, many steps, has now been tested here.
- **LOUPE instability is larger than the per-class effects being measured,** and a per-class effect is a difference of
  two learned masks. A stable mask learner is a prerequisite for any further adaptive comparison.
- Hypothesis, untested: class-specific masks help more when the reconstructor is class-aware (an interaction).
  Separate networks should lose because the VarNets are data-limited (20 → 80 volumes +1.0 dB; single-anatomy models
  lost 0.5-0.9 dB to the shared model with twice the slices), so the test should use a shared trunk with small
  per-class adapters.
- Hypothesis, untested: a previously acquired contrast of the same patient (PD → PDFS) carries the pixel-scale
  information the ACS lacks, as the pilot frame does in Wang et al.
- Hypothesis, untested: with a VarNet slope near 3 dB/octave and the census CV of 0.29, budget reallocation is worth
  about +0.16 dB on the mean.

## Caveats

- Linear SENSE on uniform lattices is not the VarNet on variable-density masks; every census dB is noise amplification.
- Low-resolution maps from a 24 x 24 centre, no prewhitening, g^2 capped at 100, slices subsampled, and the support is
  the generous trainer metric mask (knee `f` reaches 1.0 on some slices).
- Only 8 four-coil volumes; coil count is confounded with scanner model.
- The multi-step results are one seed, the dev half of knee val only, order-free, and say nothing about SPACE. The
  leak check ran on a different GPU model from the training runs; its gate (own-VarNet scores within 0.01 dB) is
  described in that branch's `kscr2/leak/README.md`.
- The multi-step snapshots are copies; this report does not recompute them and holds no checkpoint.
- Paper numbers were read from abstracts, HTML versions and two PDFs; the region-specific RL paper only from its abstract.
- The census grouping (≤8 vs >8 coils) and the disc size were chosen without a pre-registered rule.

## Figures

None.

## Next steps

1. **Mask-learner stability benchmark.** Several seeds each of LOUPE, an annealed exact-count top-k and a swap search
   that accepts only measured improvements, through one frozen VarNet; score seed-to-seed mask overlap and PSNR spread.
2. **Rate-distortion census (eval-only).** Per-volume PSNR at R = 4, 6, 8, 12, 16 through CG-SENSE with ecalib maps and
   one multi-R VarNet: the real slope spread, the oracle reallocation gain on the mean and the worst decile, and how
   well `f`, coils and noise predict it.
3. **2 x 2 specialisation test on tier L.** {shared mask, per-class frozen masks} x {shared VarNet, trunk + per-class
   adapters} with a shuffled class-to-adapter twin; the interaction is the hypothesis.
4. **Regime sweep.** The same shared-versus-adaptive comparison with 1D lines and with 1, 2, 4, 8, 16 virtual coils,
   plus a same-patient PD → PDFS oracle-correlation check (tier L has 54 pairs).

## Reproduce

`bash reports/adaptivity_levers_r8/run.sh` from the repo root. It needs the local-v1 validation volumes on
`/export/scratch`, the project `.venv` and CPU only (about 15 minutes with 160 workers); no GPU and no checkpoint. The
multi-step snapshots need commit `c718a3d` in the repository. Check with:

    python scripts/report_manifest.py check reports/adaptivity_levers_r8

## Sources

- L2SR: https://arxiv.org/abs/2212.02190v3
- ASMR: https://arxiv.org/html/2406.04318v1
- Hong et al., ECCV 2024: https://arxiv.org/html/2409.11738v2
- Wang, Noll & Fessler: https://arxiv.org/abs/2302.13468
- Region-specific RL: https://research.itu.edu.tr/en/publications/a-reinforcement-learning-approach-for-optimized-mri-sampling-with/
- DuDReTLU-net: Comput. Med. Imaging Graph. 106 (2023) 102206, doi:10.1016/j.compmedimag.2023.102206
