# Does a smaller VarNet beat 12 cascades on 20 brain volumes, and does the best size grow with data? (2026-11-14)

| | |
|---|---|
| Code | results produced at `e81deae` (run state: brain_varnet_scaling summary adds 6-vs-12 and cost); branch `exp/brain_varnet_scaling` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 21 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 102 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |

**Status: merged.** Verdict (user, 2026-11-14): merge, and follow up with a cascades × channels sweep at 80 volumes. The trainer-val subset difference (data audit below) is stated, not corrected: it does not change the ranking, and all full-val scores use the whole val set.

- **Branch:** `exp/brain_varnet_scaling`.
  - Runs trained at `35f8f2f` (c12_v40_s42 and c12_v80_s42, no early stopping) or `8485567` (the other 19, early
    stopping with patience 5).
  - Scored at `61e5876`; summary at `e81deae`.
- **Proposal:** `reports/brain_varnet_scaling/proposal.md` (frozen at `a8edd67`).

**Proposal vs outcome** (full brain val PSNR at `best_model.pt`):

| Frozen criterion | Outcome | Met? |
|---|---|---|
| Validity: every run finishes, brain SSIM ≥ 0.80 | 21/21 finished; SSIM 0.946–0.957 | met |
| Q1 at 20 volumes: smaller wins / capacity not the limit / bigger wins / inconclusive | best smaller (6) − 12: s42 **−0.24** dB [−0.33, −0.15], s43 **+0.45** dB [+0.32, +0.58]; opposite signs | **inconclusive** |
| Q2: best cascade count at 80 volumes > at 20 volumes in both seeds | s42: 12 at 20 volumes → 6 at 80; s43: 6 → 12 | **no shift** |
| Expected: 20 volumes, 4–8 cascades 0.1–0.3 dB above 12 with a smaller train − val gap | s42: all smaller sizes below 12 (−0.24 to −0.81); s43: 6 is 0.45 above. The gap is larger for 6 than for 12 in both seeds (2.30 vs 1.18, 2.33 vs 1.93 dB) | not met |
| Expected: 80 volumes, 12 cascades best or tied | 6 − 12 = +0.05 (s42), −0.02 (s43); two-seed mean +0.02 [−0.01, +0.05] | met (tied) |
| Expected: 12-cascade gain 20 → 80 volumes, +0.8 to +1.3 dB | +0.84 (s42), +1.39 (s43), mean +1.11 | met on the mean; s43 slightly above the range |

**Frozen next step:** Q1 is not "smaller wins", so go to full data with 12 cascades.

**Recommendation: merge.**
- The question is answered: within this grid, model size is not what limits brain VarNets on 20–80 volumes, and data
  is.
- The branch adds two opt-in options worth keeping on main: `datasets.val_anatomies` and
  `training.early_stop_patience` (default 0, behaviour unchanged), each with tests.
- For the follow-up, consider 6 cascades instead of 12. It ties 12 at 40 and 80 volumes with half the parameters and
  0.67× the step time. That choice is the user's; the frozen rule says 12.

## Question
Brain-only VarNets trained on 20 volumes overfit. In `reports/brain_static_vs_loupe/report.md`'s 130-epoch runs, val
peaks around epoch 60–65 and then falls while train loss keeps dropping. Two questions:
1. Does a smaller VarNet (fewer cascades) reach a higher val peak than the 12-cascade default at 20 volumes?
2. Does the best cascade count grow from 20 to 40 to 80 training volumes?

Conventions:
- PSNR / SSIM / NMSE are scored under the trainer's apodized brain metric mask.
- "Peak" means `best_model.pt`, the checkpoint with the lowest val L1 loss on the trainer's 150-slice brain val.
- Differences are A − B in dB, paired per slice, with 95% bootstrap CIs over volumes.

## Runs compared
All runs share this setup:
- frozen mask S (`loss_domain/runs_v2/loss_domain_rss_r8_flair_knee_100ep_seed42/best_model.pt`);
- R=8 of acquired; brain-only training; trainer val pinned to brain (`datasets.val_anatomies=[brain]`, 10 volumes /
  150 slices);
- training data: brain FLAIR of the local v1 manifest, tier S. 100% is 80 volumes / 1,211 slices; 20 and 40 volumes
  are fixed nested subsets (draw seed 1234);
- Muon + LayerNorm, L1 on RSS, batch 16, lr 1e-3 dropping 10× at 90% of the budget, `recon.chans=18`;
- 7,600 optimiser steps: 400 / 200 / 100 epochs at 20 / 40 / 80 volumes, val every 380 steps.

Checkpoints: `~/data/seqmri_checkpoints/brain_varnet_scaling/c<cascades>_v<volumes>_s<seed>`, linked as
`checkpoints/bvs_*`.

| Group | Cascades | Volumes | Seeds | Runs | Full-val PSNR range (dB) | SSIM range |
|---|---|---|---|---|---|---|
| seed-42 grid | 2, 4, 6, 8, 12 | 20, 40, 80 | 42 | 15 | 35.52–37.23 | 0.946–0.957 |
| seed-43 picks | 6, 12 | 20, 40, 80 | 43 | 6 | 35.76–37.15 | 0.949–0.956 |

Seed-43 picks follow the frozen rule: the two highest seed-42 peaks per data level.
- 20 volumes: {12, 6}; 40 volumes: {6, 12}; 80 volumes: {6, 12}.
- 12 cascades were already among the 20-volume picks, so no extra control run was needed.

Model cost, measured on the 80-volume seed-42 runs (A6000, one run per GPU):

| Cascades | 2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|
| Parameters (M) | 5.7 | 10.6 | 15.5 | 20.5 | 30.3 |
| Seconds per step (median) | 0.57 | 0.70 | 0.82 | 0.96 | 1.21 |

## Methods
1. **Training.** `scripts/launch/queue_brain_varnet_scaling.sh` fills a GPU queue on GPUs 0–1.
   - Early stopping (`training.early_stop_patience=5`) ends a run after 5 validations in a row without a new best val
     loss. It was added after the first two runs had started; see Caveats.
2. **Full-val and train scoring** (`scripts/eval/brain_varnet_scaling_eval.py`).
   - Scores `best_model.pt` and `last_model.pt` of every run on two sets:
     - the full brain val set (40 volumes, 605 slices);
     - the run's own training slices (20 / 40 / 80 volumes), for the train − val gap.
   - S's brain map is thresholded with the common per-slice Bernoulli draws of `eval_adaptive_samplers_r8` (seeds 1234
     and 4321), so every model sees identical masks on each slice.
3. **Summary and criteria** (`scripts/eval/brain_varnet_scaling_summary.py`).
   - Mean PSNR / SSIM / NMSE per run, the seed-43 picks, the Q1 and Q2 verdicts, 6 − 12 cascades at every data size,
     and parameters and step time.
   - Volume bootstrap: resample the 40 val volumes 10,000 times and take the slice-weighted mean difference.
4. **Trainer-val curves** (`scripts/plots/brain_varnet_scaling_trainer_val.py`): the 150-slice brain val PSNR at each
   validation, parsed from `train_sequential.log`. It is provisional and shows the curve shape only.

## Results

### Full-val PSNR at the peak checkpoint (dB)

| Volumes | c2 s42 | c4 s42 | c6 s42 | c8 s42 | c12 s42 | c6 s43 | c12 s43 |
|---|---|---|---|---|---|---|---|
| 20 | 35.63 | 35.52 | 36.09 | 36.03 | **36.33** | **36.21** | 35.76 |
| 40 | 36.10 | 36.31 | **36.81** | 36.66 | 36.68 | 36.69 | **36.78** |
| 80 | 36.23 | 37.03 | **37.23** | 37.12 | 37.17 | 37.13 | **37.15** |

- 2 and 4 cascades are clearly worse at every data size (0.2–1.0 dB below 6).
- From 6 cascades up the sizes sit within 0.3 dB of each other, and which one is best changes with the seed.

### 6 minus 12 cascades, paired per slice (dB, 95% CI over volumes)

| Volumes | s42 peak | s43 peak | Two-seed mean, peak | s42 last | s43 last |
|---|---|---|---|---|---|
| 20 | −0.24 [−0.33, −0.15] | +0.45 [+0.32, +0.58] | +0.10 [+0.01, +0.19] | +0.22 | +0.67 |
| 40 | +0.13 [+0.06, +0.21] | −0.09 [−0.16, −0.04] | +0.02 [−0.01, +0.05] | +1.08 | +0.02 |
| 80 | +0.05 [+0.03, +0.08] | −0.02 [−0.06, +0.02] | +0.02 [−0.01, +0.05] | +0.16 | +0.11 |

- Each seed's CI excludes 0 at 20 and 40 volumes, yet the sign flips between seeds. So the per-slice CI reflects
  slice sampling only, not training noise.
- At 40 and 80 volumes, 6 and 12 cascades tie (two-seed mean +0.02 dB).
- At 20 volumes the evidence leans slightly to 6 (+0.10 dB mean) but fails the frozen two-seed rule.
- At the last checkpoint, 6 is ahead or tied in all six cells. 12 cascades lose more after their peak, although last
  checkpoints come at unequal steps (Caveats).

### Seed spread of the peak (|s42 − s43|, dB)

| Volumes | 6 cascades | 12 cascades |
|---|---|---|
| 20 | 0.12 | **0.57** |
| 40 | 0.12 | 0.10 |
| 80 | 0.09 | 0.02 |

12 cascades on 20 volumes is the noisiest cell, and that noise alone decides Q1's sign.

### Data effect (peak PSNR, mean over both seeds)

| Cascades | 20 vols | 40 vols | 80 vols | 20 → 80 |
|---|---|---|---|---|
| 6 | 36.15 | 36.75 | 37.18 | +1.03 |
| 12 | 36.05 | 36.73 | 37.16 | +1.11 |

Quadrupling the data adds about 1.0–1.1 dB, against a two-seed 6 − 12 difference of at most 0.02 dB at 40–80 volumes.

### Train − val gap (dB, own train slices minus full val)

| Run | 20 vols, peak / last | 40 vols, peak / last | 80 vols, peak / last |
|---|---|---|---|
| c6 s42 | 2.30 / 5.70 | 1.78 / 2.73 | 0.80 / 1.10 |
| c12 s42 | 1.18 / 6.69 | 1.27 / 6.43 | 1.11 / 1.81 |
| c6 s43 | 2.33 / 5.57 | 1.32 / 2.05 | 0.80 / 1.04 |
| c12 s43 | 1.93 / 7.46 | 1.32 / 2.64 | 1.55 / 2.03 |

- At the peak, 6 cascades do not have the smaller gap at 20 volumes, contrary to the proposal.
- After the peak, every 20-volume run overfits by 5.6–7.5 dB.

### Trainer-val curves (150 slices, provisional)
- At 20 volumes every run peaks within steps 1,140–1,900 of 7,600 and early-stops by step 4,180.
- At 80 volumes the runs peak at steps 4,180–5,700.
- Full table: `figures/trainer_val/trainer_val_peaks.csv` (s42) and `figures/trainer_val_s43/trainer_val_peaks.csv`.

### Why val loss sits above train loss: the trainer's val subset is harder (data audit, post hoc)
`scripts/diagnostics/brain_trainval_data_audit.py` scored learning-free reconstructions under the same mask S on three
sets: the 80 training volumes, the trainer's 10-volume val subset and the full 40-volume val set.
- Zero-filled RSS, and CG-SENSE (10 iterations, BART ecalib maps).
- Also the fully sampled SENSE combination, and two trained peaks (c6_v80_s42, c12_v20_s42).

| Ratio to train | Noise σ | SNR | CG-SENSE L1 (full image) | CG-SENSE NMSE | CG-SENSE PSNR | c6_v80 L1 | c6_v80 PSNR |
|---|---|---|---|---|---|---|---|
| Trainer val (10 volumes) | 1.21× | 0.90× | 1.29× | 1.53× | −0.89 dB | 1.33× | −0.86 dB |
| Full val (40 volumes) | 1.04× | 0.98× | 1.07× | 1.10× | −0.57 dB | 1.13× | −0.80 dB |

- **The subset itself causes most of the early gap.** A reconstruction with no training already has 1.26–1.29× higher
  L1 on the trainer's val subset than on train. That matches the 1.2–1.5× val/train loss ratio at the first validation.
- **Why the subset is harder: its three 1.5 T volumes.**
  - 1.5 T (all Siemens Aera) is 24% of train volumes, 25% of full val and 30% of the subset, so the share is close.
  - But the subset's three 1.5 T volumes are all AXFLAIR_202, with SNR 12.4 against 14.9 for train's 1.5 T volumes.
    c6_v80 scores 33.6 dB on them, against 36.5 dB on train's 1.5 T.
  - Its 3 T volumes match train: SNR 23.8 vs 24.7, c6_v80 38.8 vs 38.5 dB.
  - Within each field strength, train → full val costs c6_v80 0.73 dB (3 T) and 0.89 dB (1.5 T), and CG-SENSE 0.48 dB
    and 0.65 dB.
- **Normalization equalizes brightness, not noise.** The loader divides each slice by the peak of its centre-10×10
  k-space image, so mean signal in the brain mask is about 0.62 in every set, while noise σ varies about 2× between
  groups.
- **The training L1 includes the background.** It averages over the whole image, and about 62% of pixels lie outside
  the metric mask, where the target is noise. So low-SNR slices raise the loss even at equal reconstruction quality.
- **What is left at 80 volumes:** on the representative full val, c6_v80 loses 0.80 dB from train to val, while
  learning-free reconstructions lose 0.22 dB (zero-filled) to 0.57 dB (CG-SENSE). Only a small part of the 80-volume
  gap is fitting the training set.
- Noise σ is the std of the outer 16 readout lines inside each slice's acquired extent. Short-readout volumes (512,
  groups 203/206 and some 202) are zero-padded to the grid. The phase-encode edges give the same ratios (1.16×
  trainer val, 1.03× full val). Results, with field strength and scanner per slice, are in `data_audit/`.

## Conclusions
- **Q1 is inconclusive.** At 20 volumes the best smaller size (6) is −0.24 dB below 12 in seed 42 and +0.45 dB above it
  in seed 43. The swing comes from 12 cascades, whose peak moves 0.57 dB between seeds.
- **Q2: no shift.** No size wins at 80 volumes. 6, 8 and 12 cascades lie within 0.1 dB (s42), and 6 − 12 is
  −0.02 dB in s43.
- **The gain from cascades levels off around 6.** 2–4 cascades lose 0.2–1.0 dB (seed 42 only). 6, 8 and 12 sit within
  0.3 dB in seed 42, and 6 vs 12 ties in the two-seed mean at every data size (+0.02 to +0.10 dB). The 2-, 4- and
  8-cascade results are single-seed.
- **Data is the lever.** 20 → 80 volumes adds +1.0 to +1.1 dB (two-seed mean), within the predicted +0.8 to +1.3 dB.
- **Overfitting at 20 volumes is a data problem, not a capacity problem in this range.** All sizes peak by step
  ~1,900 and then lose 0.35–1.34 dB of trainer-val PSNR. Smaller models peak no higher.
- Hypothesis, untested: 12 cascades are more seed-sensitive at 20 volumes because they overfit faster, so the
  checkpoint picked on 150 slices lands at different points of a falling curve.

## Caveats
- **Early stopping was added mid-grid.** The user asked for it after the seed-42 grid had started.
  - c12_v40_s42 and c12_v80_s42 trained without it (commit `35f8f2f`) and ran the full 7,600 steps. The other 19 runs
    used patience 5 (`8485567`).
  - The peak checkpoint does not depend on this, since `best_model.pt` is the val-loss minimum either way.
  - `last_model.pt` does. For early-stopped runs it sits 5 validations after the best; for full runs it sits at step
    7,600. Last-checkpoint comparisons are therefore not at equal steps.
  - c8_v80_s42 and c12_v80_s43 ran the full budget even with early stopping on.
- **Two seeds only, and only 6 and 12 cascades got seed 43.** The 2-, 4- and 8-cascade numbers are single-seed.
- **The peak is chosen on 150 slices.** `best_model.pt` minimises L1 val loss on the trainer's 10-volume val. In some
  runs the last checkpoint beats it on full val (c4_v20_s42 +0.38, c4_v40_s42 +0.39, c2_v80_s42 +0.17 dB).
- **Seed-43 crashes.** Four seed-43 jobs crashed on out-of-memory: leftover seed-42 workers put two runs on one GPU.
  All four were rerun in full from scratch.
- **Channels were not varied.** `recon.chans=18` is fixed; width scaling is a separate follow-up.
- **Frozen mask.** Every run uses mask S. Jointly learned masks may change the best size.
- **Step times are from one run per size.** They come from the shared server, with one run per GPU.

## Figures
- `figures/full_val_vs_cascades.png`: seed-42 full-val PSNR at the peak against cascades, one line per data size.
  `figures/full_val_vs_cascades_s43.png` is the same for seed 43 (6 and 12 cascades only).
- `figures/full_val_vs_volumes.png` and `figures/full_val_vs_volumes_s43.png`: the same data against training volumes,
  one line per cascade count.
- `figures/trainer_val/trainer_val_curves.png` and `figures/trainer_val_s43/trainer_val_curves.png`: 150-slice trainer
  val PSNR against step. Lines end where a run stopped early.
- `figures/trainer_val*/train_loss_curves.png`: train loss against step.
- `figures/trainer_val*/peak_vs_cascades.png` and `figures/trainer_val*/peak_vs_volumes.png`: trainer-val peaks.

## Next steps
1. Follow the frozen rule:
   - use full data with the size chosen in the verdict, 12 cascades per the rule or 6 as the cheaper tie. Here "full
     data" means all 80 brain FLAIR volumes of the local tier S (`datasets.limit_volumes=1.0`, 1,211 slices). More
     FLAIR exists only in tier L on CoreHPC (220 more volumes); the 80 local T2 volumes are a different contrast;
   - then rerun static vs learned masks (S, M, U) there.
2. Optional: a third seed for 6 and 12 cascades at 20 volumes would settle Q1. It costs about 2 GPU-h.
3. Width scaling (`recon.chans` 12 / 18 / 32) at 6 cascades and 80 volumes.
4. Pick `best_model.pt` and early-stop on the full 40-volume val, or a subset stratified by acquisition group. The 10-volume subset over-represents low-SNR AXFLAIR_202, so it is harder and noisier than both train and full val.

## Reproduce
```bash
bash reports/brain_varnet_scaling/run.sh
python scripts/report_manifest.py check reports/brain_varnet_scaling
```
- Needs the 21 checkpoints in `~/data/seqmri_checkpoints/brain_varnet_scaling`, linked as `checkpoints/bvs_*`, the
  fastMRI brain data of the local v1 manifest, and one GPU.
- Training: `scripts/launch/queue_brain_varnet_scaling.sh`.
