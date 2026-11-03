# Does a complex-image or k-space training loss beat RSS-image L1 for R=8 LOUPE + VarNet? Round 2 (2026-11-02)

| | |
|---|---|
| Code | results produced at `82de7c4` (loss_domain: let intensity_fit.py take a runs root and arm list (round 2)); branch `exp/loss_domain` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 4 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 25 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |

- **Status:** awaiting the human verdict.
- **Branch:** `exp/loss_domain`. Runs trained at `676675b` (clean tree); results produced at `82de7c4`.
- **Seeds:** seed 42 only. The frozen criteria ask for seeds 42-44. On 2026-11-02 the user dropped seeds 43-44
  because the effect sizes are large. That covers the kspace_gt and complex verdicts. rss_M's +0.07 dB stays
  inconclusive at one seed.

### Proposal vs outcome

The criteria were frozen in `proposal.md` on 2026-10-30 and amended for round 2 on 2026-11-01. Results are seed 42 on
all 1938 val slices.

| Criterion | Result | Status |
|---|---|---|
| Loss bar: rss_M − rss ≥ +0.10 dB PSNR_Masked, positive in every seed | +0.071 dB | not met (inconclusive at one seed) |
| Loss bar: kspace_gt − rss | −2.393 dB (95% CI over volumes −3.01 to −1.79) | not met |
| Loss bar: complex − rss | −8.692 dB (−9.09 to −8.31) | not met |
| Mask bar: \|rss_M − rss\| ≥ 0.10 dB, same sign in every seed | +0.071 dB | not met (inconclusive at one seed) |
| Health gate: per condition, SSIM_Masked no more than 0.01 below the rss twin | rss_M +0.001 in every condition; kspace_gt brain −0.004, PD −0.001, **PDFS −0.025**; complex −0.100 to −0.109 | rss_M pass; kspace_gt fail (PDFS); complex fail (expected) |
| A complex miss is checked against the pseudo-output SSIM (0.837 / 0.873 / 0.669, brain / PD / PDFS) | 0.782 / 0.810 / 0.674 | below the pseudo-output in brain and PD (pseudo values from the round-1 loader and cache) |
| AGENTS.md: SSIM_Masked < 0.80 means a bug | rss 0.860, rss_M 0.861, kspace_gt 0.850, **complex 0.755** | complex below; its cause was not re-diagnosed on this run |
| Twin within 0.01 of the old control's 0.839 (informational since round 2) | 0.860 | not near; the old control had Ω on |
| LOUPE mask correlation with the twin inside the rss seed-to-seed range (reported, no gate) | 0.83 / 0.64 / 0.61 (rss_M / kspace_gt / complex) | not evaluable: one rss seed |

| Arm | Expected (amendment 2026-11-01) | Observed |
|---|---|---|
| rss | slightly above round 1 (32.1 dB / 0.836), mostly in knee; brain about ±0.1 dB | 33.62 dB / 0.860: +1.7 brain, +1.8 PD, +1.1 PDFS |
| rss_M | within ±0.1 dB of rss | +0.07 dB |
| kspace_gt | sign unknown; darker than t in low-SNR tissue, PDFS most | −2.39 dB, PDFS −4.88; output mean 0.87 of t in PDFS; ties rss after an oracle gain+offset fit |
| kspace_gt data consistency | stays at rss's level (19-50% error at sampled points) | not measured |
| complex | well below rss; health-gate miss expected | −8.69 dB, SSIM −0.105 |
| complex phase std | falls toward 0 | 0.431 rad vs 0.485 for rss |

### Recommendation

**Park** the branch. No loss passes the loss bar, so RSS-image L1 stays the default. Two items from this branch are
worth acting on separately (Next steps 1-2): the brain-flip loader fix `b25d2c5`, and the +1.5 dB control gain that came
with the sensitivity-net support mask turned off.

## Question

Production training minimises L1 between the scored RSS magnitude and the RSS target. That leaves the maps' common
phase free and scores only magnitude. Round 2 tests three alternatives against the RSS-L1 twin of the same seed:
- the same L1 restricted to the ESPIRiT support M;
- an L1 on coil k-space against the measured, fully sampled k-space;
- an L1 on the complex image against the ESPIRiT image.

Each is checked for changes to the scored RSS metrics, the learned LOUPE mask and the maps' phase. Every metric is
taken inside the apodized metric mask against the RSS target. PSNR is in dB. R=8 counts samples within each slice's
acquired k-space extent.

Round 1 (2026-10-30/31; `eval/summary.json`, `diag/`) found three defects:
- a noise-floor pedestal in the k-space RSS loss;
- mixed ESPIRiT and learned-map phase conventions in the complex and SENSE k-space targets;
- an off-by-one brain flip in the loader.

Its numbers are not reused. The amendment in `proposal.md` gives details.

## Runs compared

Notation:

| Symbol | Meaning |
|---|---|
| `t` | RSS target |
| `rss` | scored output, projected onto the acquired extent |
| `x̂` | VarNet complex image |
| `S` | learned maps |
| `E` | acquired k-space extent |
| `F` | centred FFT |
| `k` | measured coil k-space |
| `x_E` | cached ESPIRiT image (BART ecalib, v2 cache) |
| `M` | `\|x_E\| > 0` |

| Arm | Loss (`loss.loss_type`) | Run under `checkpoints/` | wandb | PSNR_Masked | SSIM_Masked |
|---|---|---|---|---|---|
| rss (control) | `l1`: mean \|rss − t\| | `loss_domain_rss_r8_flair_knee_100ep_seed42` | nle6f5hs | 33.62 | 0.860 |
| rss_M | `l1` + `loss.support_mask=true`: mean \|M·rss − M·t\| | `loss_domain_rss_M_r8_flair_knee_100ep_seed42` | kta13bb1 | 33.69 | 0.861 |
| kspace_gt | `kspace_gt_l1`: mean \|(F(S·x̂) − k)·E\| | `loss_domain_kspace_gt_r8_flair_knee_100ep_seed42` | 9nvbuqj0 | 31.23 | 0.850 |
| complex | `complex_l1`: mean \|x̂_E − x_E\| | `loss_domain_complex_r8_flair_knee_100ep_seed42` | epp867rr | 24.93 | 0.755 |

The `checkpoints/` entries link to `~/data/seqmri_checkpoints/loss_domain/runs_v2/`.

Shared settings:
- **Data:** local v1 FLAIR + knee-PD + knee-PDFS at 320×320. Train is 80 / 40 / 40 volumes (3929 slices); val is
  40 / 20 / 20 volumes (1938 slices).
- **Sampler:** shared `loupe_2d`, straight-through, VD start p=4, no anatomy conditioning. R=8 of the acquired
  k-space, `coil_combine=sense`.
- **Reconstructor:** E2E VarNet, 12 cascades, `SensetivityModel_mc` maps. The sensitivity-net support mask Ω is off
  (`+sampler.sense_mask_method=null`), and `csme=false`.
- **Training:** batch 16, 100 epochs, lr 1e-3 stepped ×0.1 at epoch 90, seed 42. Arms share the init, the batch order
  and the mask draws.
- **Compute:** about 12.7 h per run on one A6000.

Round-1 control, for context: 32.08 dB / 0.835. It had Ω on and the old brain flip (`eval/summary.json`, trained at
`1ba1ccf`). Its config differs from round 2's rss only in `sense_mask_method`, and its code differs only by the flip fix
and the new loss options.

## Methods

1. **Training.** `scripts/launch/queue_loss_domain_local.sh` runs `scripts/launch/train_loss_domain_r8_local_v1.sh`
   for each arm.
2. **Scoring** (`scripts/eval/eval_loss_domain.py`, output `eval_v2/summary.json`, `eval_v2/eval.log`).
   - Each run's `last_model.pt` is scored on all 1938 val slices with two fixed mask draws (torch seeds 1234 and
     4321), averaged per slice.
   - Metrics are PSNR_Masked = 10·log10(max(t)² / masked MSE), NMSE and SSIM_Masked, all inside the apodized metric
     mask.
   - The paired difference is arm minus rss, per slice. It is reported as the mean over slices with a 95% bootstrap
     CI over volumes (2000 resamples of the 80 volumes, slice-weighted). That CI covers slice and volume variability
     within one trained model. It does not cover training-seed spread, which is 0.3-0.5 dB.
   - The LOUPE mask correlation is the Pearson correlation of the learned probability maps, sigmoid(slope·w).
   - Maps' phase is measured on the angle of Σ_c conj(S_c)·coil_c·conj(x_E), over metric mask ∩ map support ∩ M:
     - its circular std, sqrt(−2·ln R), where R is the |x_E|-weighted mean resultant length;
     - the share of neighbouring pixel pairs where that angle jumps by more than π/2.
3. **Intensity diagnostic** (`reports/loss_domain/diag/intensity_fit.py`, output `diag/intensity_fit_v2.txt`).
   - It uses every 12th val slice (162) and one mask draw (seed 1234).
   - Per slice, inside the metric mask, it computes PSNR raw, after a least-squares gain, and after a least-squares
     gain+offset. Both fits are to the target, so they are oracle bounds, not scores.
   - It also gives the output/target mean ratio, and the share of the masked error's energy at k-space radius < 16 px.
4. **Pre-launch mask diagnostics**, which led to Ω being turned off: `reports/loss_domain/diag/mask_overlap.py`
   (`diag/mask_overlap.txt`) and `reports/loss_domain/diag/plot_masks.py` (`diag/masks.png`).

## Results

### Scored metrics (all 1938 val slices, PSNR_Masked dB / SSIM_Masked)

| Arm | All | Brain | Knee-PD | Knee-PDFS | NMSE, all |
|---|---|---|---|---|---|
| rss | 33.62 / 0.860 | 32.83 / 0.882 | 35.57 / 0.916 | 32.36 / 0.783 | 0.0054 |
| rss_M | 33.69 / 0.861 | 32.92 / 0.883 | 35.67 / 0.917 | 32.39 / 0.784 | 0.0055 |
| kspace_gt | 31.23 / 0.850 | 31.36 / 0.878 | 34.79 / 0.915 | 27.48 / 0.758 | 0.0161 |
| complex | 24.93 / 0.755 | 25.57 / 0.782 | 25.95 / 0.810 | 23.30 / 0.674 | 0.0421 |
| round-1 rss (Ω on, old flip) | 32.08 / 0.835 | 31.09 / 0.842 | 33.77 / 0.895 | 31.27 / 0.768 | — |

Every round-2 arm except complex is above the 0.80 SSIM floor. The ranking is the same in every condition: rss_M ≈ rss >
kspace_gt > complex.

### Paired differences against rss (PSNR_Masked dB, 95% CI over volumes, share of volumes where the arm is higher)

| Pair | All | Brain | Knee-PD | Knee-PDFS |
|---|---|---|---|---|
| rss_M − rss | +0.071 [+0.048, +0.088], 95% | +0.094 [+0.080, +0.108], 95% | +0.097 [+0.078, +0.117], 95% | +0.022 [−0.040, +0.058], 95% |
| kspace_gt − rss | −2.393 [−3.014, −1.792], 15% | −1.465 [−1.907, −1.072], 10% | −0.784 [−1.316, −0.265], 40% | −4.884 [−5.924, −3.897], 0% |
| complex − rss | −8.692 [−9.086, −8.307], 0% | −7.253 [−7.693, −6.808], 0% | −9.622 [−10.387, −8.903], 0% | −9.062 [−9.477, −8.663], 0% |

SSIM_Masked differences against rss:

| Pair | All | Brain | Knee-PD | Knee-PDFS |
|---|---|---|---|---|
| rss_M − rss | +0.0011 | +0.0012 | +0.0010 | +0.0011 |
| kspace_gt − rss | −0.0101 | −0.0039 | −0.0011 | −0.0250 |
| complex − rss | −0.1051 | −0.1003 | −0.1059 | −0.1086 |

rss_M's gain is consistent across volumes but smaller than the 0.10 dB bar and the seed spread. kspace_gt's deficit sits
mostly in PDFS. complex is far below rss everywhere.

### Intensity diagnostic (162-slice subset, one mask draw; the fits are oracle bounds)

| Arm | Condition | Raw | Gain | Gain+offset | Output/target mean | Error energy at k < 16 px |
|---|---|---|---|---|---|---|
| rss | all | 33.80 | 33.85 | 33.93 | 0.997 | 0.044 |
| rss | brain / PD / PDFS | 33.27 / 35.35 / 32.70 | 33.29 / 35.41 / 32.79 | 33.31 / 35.57 / 32.85 | 0.999 / 0.996 / 0.995 | 0.035 / 0.059 / 0.036 |
| rss_M | all | 33.86 | 33.92 | 34.01 | 0.996 | 0.043 |
| kspace_gt | all | 31.50 | 32.10 | 33.91 | 0.933 | 0.350 |
| kspace_gt | brain / PD / PDFS | 31.95 / 34.56 / 27.97 | 32.30 / 34.79 / 29.17 | 33.44 / 35.80 / 32.43 | 0.959 / 0.971 / 0.870 | 0.259 / 0.228 / 0.559 |
| complex | all | 25.32 | 26.23 | 27.73 | 0.902 | 0.597 |
| complex | brain / PD / PDFS | 26.11 / 26.31 / 23.57 | 27.27 / 26.77 / 24.73 | 28.03 / 27.52 / 27.67 | 0.943 / 0.948 / 0.818 | 0.467 / 0.589 / 0.726 |

kspace_gt's output is 7% darker than the target overall (13% in PDFS), and 35% of its error energy lies at low
frequency, against 4% for rss. After the oracle gain+offset fit it ties rss: 33.91 vs 33.93 dB overall, PDFS 32.43 vs
32.85. complex stays 6.2 dB below rss after the same fit. On this subset raw rss reads 33.80 dB, against 33.62 on the
full set.

### Maps' phase against ESPIRiT, and learned masks

| Arm | Phase circular std (rad): all / brain / PD / PDFS | Share of π/2 jumps | LOUPE mask correlation with rss |
|---|---|---|---|
| rss | 0.485 / 0.416 / 0.515 / 0.517 | 0.0020 | — |
| rss_M | 0.484 / 0.416 / 0.514 / 0.516 | 0.0022 | 0.83 |
| kspace_gt | 0.485 / 0.426 / 0.509 / 0.516 | 0.0025 | 0.64 |
| complex | 0.431 / 0.336 / 0.470 / 0.480 | 0.0157 | 0.61 |

kspace_gt trains the maps on the measured data, but their phase relative to ESPIRiT stays at rss's level. complex lowers
the std only from 0.485 to 0.431 rad, and has 8× as many phase jumps. The mask correlations have no rss seed-to-seed
reference, so their size can't be judged.

## Conclusions

- **No alternative loss beats RSS-image L1.** The loss bar is missed by all three arms (results tables 1-2).
  kspace_gt (−2.4 dB) and complex (−8.7 dB) miss by far more than the 0.3-0.5 dB training-seed spread, so one seed
  settles them.
- **rss_M is a null at this precision.** It is +0.07 dB with SSIM +0.001, inside the expected ±0.1 dB and below the
  0.10 dB bar. One seed can't separate it from training noise, so there is no case for masking the image loss with M.
- **kspace_gt, as scored, is worse than rss and fails the health gate in PDFS** (SSIM −0.025), so under the frozen
  rule its numbers are not trusted as a reconstruction-quality comparison. The deficit is a low-frequency intensity
  offset: the output is darker, the error energy sits at low k, and the arm ties rss after an oracle gain+offset fit.
  Hypothesis, untested: the cause is the RSS noise bias, which the measured (linear) k-space target lacks and the RSS
  target carries.
- **complex is broken, not just offset.** Its SSIM of 0.755 is under the AGENTS.md floor, and it stays 6.2 dB below rss
  after an oracle gain+offset fit. Hypothesis, untested on these checkpoints: the cause is the ESPIRiT phase convention
  that round 1 diagnosed (BART run without `-P`; `diag/complex_phase.py`, round-1 checkpoint). The diagnostic was not
  rerun on the round-2 checkpoint, so a bug is not excluded.
- **The control improved by 1.5 dB over round 1** (32.08 → 33.62 dB; brain +1.7, PD +1.8, PDFS +1.1), and SSIM went
  from 0.835 to 0.860. Two changes are confounded: Ω off, and the brain flip fix. Hypothesis, untested: Ω off drives
  most of the knee gain, because knee data never had the flip bug. The model is shared across anatomies, though, so
  the brain fix may also affect knee.

## Caveats

- **Seeds:** one training seed per arm (user waiver, 2026-11-02). The CIs are within one trained model, over volumes.
- **Scoring target:** kspace_gt and complex are scored against the RSS target, whose noise bias they don't model. The
  oracle fit is only a diagnostic upper bound on what a calibration could recover, and it was run on a 162-slice
  subset with one mask draw.
- **complex:** not re-diagnosed on the round-2 checkpoint. The pseudo-output SSIM reference (0.837 / 0.873 / 0.669)
  was computed with the round-1 loader and cache.
- **Missing readouts:** kspace_gt's data consistency at the sampled points (expected 19-50%, as for rss) was not
  measured. The mask correlations have no rss seed-to-seed reference.
- **Round-1 comparison:** it spans two code commits (`1ba1ccf` → `676675b`). For the l1 arm, the only relevant
  differences are Ω and the brain flip. The brain metric mask and targets shift by one row between rounds.
- **Untracked inputs:** the checkpoints are untracked. Gitignored eval caches (`eval*/runs/`) and figures were moved to
  `~/data/seqmri_checkpoints/loss_domain/report_untracked/`, and `run.sh` regenerates the round-2 ones.
- **Script location:** the diagnostic scripts live in `reports/loss_domain/diag/`, not `scripts/`.
- **Round-1 files** in this folder (`eval/`, `qc/`, the other `diag/` scripts and outputs, `eval_seed42.log`) were
  produced at earlier commits. `run.sh` does not regenerate them. `scripts/eval/eval_loss_domain.py` now lists the
  round-2 arms; the round-1 eval is at `88d7f8c`.

## Figures

- `diag/masks.png` (gitignored; `run.sh` regenerates it, and a copy is in
  `~/data/seqmri_checkpoints/loss_domain/report_untracked/diag/`). It shows the metric mask, Ω and M on six val slices,
  including the holes in Ω inside the anatomy that led to turning Ω off.
- No new round-2 figures.

## Next steps

1. **Ω on vs off on the fixed loader**, on its own branch. Use the RSS L1 with matched seeds, to attribute the +1.5 dB
   control gain before changing the default `sense_mask_method`.
2. **Bring the brain flip fix `b25d2c5` to main.** Brain checkpoints trained on main score about 1 dB lower on the fixed
   loader, so plan retraining, or keep the old flip only for evaluating old checkpoints.
3. **Rerun the round-1 diagnostics on round 2.** When a GPU is free, run `diag/complex_phase.py` and
   `diag/kspace_consistency.py` on the round-2 checkpoints to confirm complex's cause and measure kspace_gt's data
   consistency.
4. **If a linear (k-space or complex) loss is still wanted:**
   - score it against a target with matching noise statistics, or report it with a gain-calibrated metric;
   - give the complex arm a phase convention the learned maps can produce, such as BART `-P` or a phase-aligned
     ESPIRiT target.

## Reproduce

Run `bash reports/loss_domain/run.sh` from the repo root. It needs:
- the four round-2 checkpoints in `~/data/seqmri_checkpoints/loss_domain/runs_v2`;
- the round-1 rss checkpoint in `.../loss_domain/runs`, for the model check in `mask_overlap.py`;
- the v2 ESPIRiT cache `/export/scratch/bkadota/fastmri_complex_espirit_v2`;
- the local v1 raw data;
- one GPU, for about 30 min.

Check the folder with `python scripts/report_manifest.py check reports/loss_domain`.
