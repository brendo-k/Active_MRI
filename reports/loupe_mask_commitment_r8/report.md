# Why does LOUPE drop the right k-space band, and which fix keeps it? (VarNet + LOUPE, R=8, 2026-11-19)

| | |
|---|---|
| Code | results produced at `287a06b` (loupe_mask_commitment_r8: mask band, criteria summary and recon figure scripts); branch `exp/loupe_mask_commitment_r8` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 10 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 8 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |
| Training commit | not recorded for these runs (they predate `run_info.json`) |

- **Training commit, corrected.** The runs have a `run_info.json`, but its `commit` is empty because the CoreHPC code copy
  had no `.git`. All 8 runs trained from run state `0bf4cec` (sbatch array 3118620). `eval_last_model.json` was
  scored at `0bf4cec`; the eval script is unchanged at `287a06b`, and the other results were regenerated at `287a06b`.
- **Branch**: `exp/loupe_mask_commitment_r8`. Results at `287a06b`; the results commit is the branch tip after this report.
- **Status**: awaiting the human verdict.

### Proposal vs outcome (criteria frozen in `proposal.md`, commit `75b14ca`)

| arm | band kept: count >= 200, both seeds | fix works: mean gain >= +0.10 dB, both > 0, no condition < −0.05, SSIM >= −0.002 |
|---|---|---|
| `clamp` | **not met** (24.4 / 13.1) | **not met** (−0.020 / −0.061, mean −0.040) |
| `anneal` | **not met** (88.6 / 348.0) | **not met** (+0.005 / +0.065, mean +0.035) |
| `mix` | **met** (539.0 / 527.3) | **not met** (−0.018 / +0.095, mean +0.039; knee-PD s42 −0.122) |

**Reading by the frozen rule: inconclusive.** `clamp` fails "band kept", which matches the race reading, but neither
`anneal` nor `mix` passes both criteria.

### Recommendation

**Park.** No arm passes the frozen criteria, and none of the three changes is worth putting on `main`. Saturation is ruled
out as the cause. Whether the band alone explains the +0.17 dB `lr4e3_step` gap is still open, because `mix` keeps the band but
also puts its own cost on the recon.

## Question

In `reports/varnet_loupe_lr_schedule_r8/report.md`, both 12-cascade lr 1e-3 controls left a band of k-space unsampled. The band
is rows 148–172 (within 12 of centre) and phase-encode columns 232 and up (more than 72 right of centre). The runs that kept
it were 0.12–0.17 dB better. This experiment asks whether the band is lost to sigmoid saturation, or to an early race in which a
young VarNet teaches the mask that the band is useless. It also asks whether either fix gains PSNR. "Band count" is the summed
LOUPE probability p = sigmoid(slope · logit) in that band, before the per-slice budget rescale (expected samples per slice).

## Runs compared

All runs use E2E VarNet 12 cascades × 18 channels with Muon + LayerNorm and LOUPE `loupe_2d` from a uniform-noise start, slope 10.
R=8 uses `budget_over=acquired`, batch 16, 100 epochs. Recon lr is 1e-3, stepped ×0.1 at epoch 90; mask lr is 1e-3.
Training data is the full local v1 FLAIR + knee set (160 train volumes), on one CoreHPC RTX Pro 6000 per run, about 3.7 h each.
Scores below are on full val, `last_model.pt`, averaged over 2 seeded mask draws.

| run | change vs control | band count | PSNR all | brain | knee-PD | knee-PDFS | SSIM all |
|---|---|---|---|---|---|---|---|
| control_s42 | none | 5.8 | 33.820 | 32.967 | 36.001 | 32.377 | 0.8633 |
| control_s43 | none | 2.9 | 33.799 | 32.940 | 36.002 | 32.340 | 0.8621 |
| clamp_s42 | soft clamp \|slope·logit\| <= 6, leak 0.01 | 24.4 | 33.800 | 32.954 | 35.965 | 32.367 | 0.8631 |
| clamp_s43 | " | 13.1 | 33.738 | 32.855 | 35.917 | 32.327 | 0.8615 |
| anneal_s42 | slope 2 → 10 over epochs 0–20 | 88.6 | 33.825 | 32.956 | 36.004 | 32.399 | 0.8625 |
| anneal_s43 | " | 348.0 | 33.864 | 33.012 | 36.077 | 32.388 | 0.8630 |
| mix_s42 | 25% of batches use a flat p = 0.5 mask at the same budget; recon-only update | 539.0 | 33.801 | 32.996 | 35.879 | 32.422 | 0.8634 |
| mix_s43 | " | 527.3 | 33.894 | 33.118 | 35.984 | 32.475 | 0.8643 |

Checkpoints: `/export/scratch/bkadota/loupe_mask_commitment_r8/runs/<run>/last_model.pt` (symlinked as `checkpoints/lmc_<run>`).
The controls were rerun at the new run state rather than reused. They reproduce the previous controls: band 5.8 / 2.9 vs
8.4 / 0.2 (re-scored here as a check, `masks.json` → `reference_lr1e3_step_12c`), and PSNR 33.82 / 33.80 vs 33.81 / 33.77.
SSIM is 0.86 overall and 0.78 on knee-PDFS. That knee-PDFS level is the known open SSIM issue, the same in every arm.

## Methods

1. **Paired eval.** `scripts/eval/varnet_loupe_lr_schedule_r8_eval.py` scores each run per slice on full local v1 val:
   80 volumes, 1,938 slices. It uses the trainer's apodized metric mask, data range = slice max, and the same 2 seeded mask
   draws for every run; the per-slice scoring is `score` in `scripts/eval/varnet_train_speed_eval.py`. Each arm is differenced against the control of the same seed. The 95% CI is over per-volume mean
   differences; "volumes better" is the share of volumes with a positive mean difference.
2. **Mask metrics.** `scripts/plots/loupe_mask_commitment_r8_masks.py` rebuilds p from each checkpoint's logits through the
   run's own `ProbMask`, so the clamp arm's soft clamp is applied. The band rectangle and support threshold come from
   `scripts/plots/varnet_loupe_lr_schedule_r8_masks.py`. It reports the band count; mean p; support (share of
   locations with p above the support threshold); and saturated share (|slope·logit| > ln 99).
3. **Criteria and curves.** `scripts/eval/loupe_mask_commitment_r8_summary.py` applies the frozen criteria and reads the
   per-anatomy trainer val PSNR from each run's log, logged every 5 epochs. Those numbers use the trainer's batch-max data
   range, sit about 4.6 dB above the eval on brain, and are compared only with each other.
4. **Recons.** `scripts/plots/loupe_mask_commitment_r8_recons.py` shows, per condition, the val slice at the median control
   PSNR (seed 43). Every arm gets the same mask-draw seed; the figure shows |error| × 5 inside the metric mask and the drawn
   mask.

## Results

### Paired PSNR vs control (dB, full val)

| arm, seed | all | 95% CI | brain | knee-PD | knee-PDFS | volumes better | SSIM diff |
|---|---|---|---|---|---|---|---|
| clamp s42 | −0.020 | [−0.027, −0.014] | −0.013 | −0.037 | −0.010 | 19% | −0.0003 |
| clamp s43 | −0.061 | [−0.073, −0.048] | −0.085 | −0.085 | −0.014 | 6% | −0.0006 |
| anneal s42 | +0.005 | [−0.016, +0.026] | −0.011 | +0.003 | +0.022 | 31% | −0.0008 |
| anneal s43 | +0.065 | [+0.054, +0.076] | +0.072 | +0.075 | +0.047 | 94% | +0.0009 |
| mix s42 | −0.018 | [−0.049, +0.013] | +0.029 | −0.122 | +0.044 | 50% | +0.0001 |
| mix s43 | +0.095 | [+0.060, +0.130] | +0.178 | −0.017 | +0.135 | 89% | +0.0022 |

`clamp` is slightly worse in both seeds. `anneal` and `mix` each have one seed that is clearly better and one that is not.
For reference, `lr4e3_step` beat this control by +0.171 dB in `reports/varnet_loupe_lr_schedule_r8/report.md`. The CIs reflect
volumes, not seeds: the seed-to-seed swing of an arm (0.04–0.11 dB) is larger than any CI half-width here.

### Mask shape at epoch 100

| run | band count | mean p | support | saturated share |
|---|---|---|---|---|
| control s42 / s43 | 5.8 / 2.9 | 0.134 / 0.138 | 0.223 / 0.224 | 0.752 / 0.754 |
| clamp s42 / s43 | 24.4 / 13.1 | 0.139 / 0.143 | 0.223 / 0.220 | 0.753 / 0.761 |
| anneal s42 / s43 | 88.6 / 348.0 | 0.139 / 0.141 | 0.219 / 0.224 | 0.762 / 0.755 |
| mix s42 / s43 | 539.0 / 527.3 | 0.144 / 0.147 | 0.238 / 0.247 | 0.716 / 0.706 |

`mix` is the only arm whose mask changed shape. Its support grew from 0.22 to 0.24–0.25 and fewer logits saturated. It also
kept a band larger than any run in the previous experiment (396–475). About 5 of each `clamp` count comes from the clamp's
probability floor (p >= sigmoid(−6) everywhere), not from sampling the band.

### Training curves

Trainer val PSNR at epoch 100 (brain / knee) is 37.60 / 32.19–32.20 for both controls, 37.52–37.58 / 32.18–32.19 for clamp,
37.59–37.67 / 32.25–32.29 for anneal and 37.66–37.81 / 32.22–32.28 for mix. On brain, `mix_s42` is the lowest of the 8 runs at
every val point from epoch 20 to 60. At epoch 20 it is 0.26 dB below the next-lowest run and 0.64 dB below the best. `mix_s43`
sits mid-pack. Both `mix` runs are still rising at epoch 100, which no other arm is (`figures/curves.png`).

## Conclusions

- **Saturation is not why the band is lost.** Keeping gradient through saturated logits (`clamp`) leaves the band empty (13–24,
  of which about 5 is the floor) and costs 0.02–0.06 dB. This matches the earlier grad-floor result (`reports/GRAD_FLOOR_R8.md`).
- **Delaying commitment or keeping the VarNet competent everywhere can keep the band.** `mix` keeps it in both seeds and
  `anneal` in one. So the loss happens early and depends on the VarNet. That fits the race / co-adaptation reading, but it
  does not pass the frozen rule.
- **Keeping the band through `mix` did not bring the gain, but this is confounded.** `mix` has the largest band of any run in
  either experiment, yet gains only −0.018 / +0.095 dB, and seed 42's knee-PD drops 0.122 dB. But `mix` also costs the recon
  something: 25% of its steps are on off-distribution masks, the mask gets fewer updates, its curves lag early, and it is still
  rising at epoch 100. A band benefit offset by that cost fits the data equally well. The cleanest case of the band without
  that cost is `anneal_s43`: band 348, +0.065 dB. That hints the band is worth something, but less than 0.17 dB, and it is one
  seed. Whether the band alone explains the `lr4e3_step` gap is unresolved. Hypothesis, untested: the better early VarNet of
  `lr4e3_step` is the cause, and the band follows from it.
- **`anneal` changes which way a seed goes, not reliably the outcome.** One seed kept the band (348) and gained +0.065 dB, better
  in 94% of volumes and every condition. The other seed lost the band (89) and gained +0.005 dB.

## Caveats

- Two seeds per arm. Seed-to-seed swings within an arm (0.04–0.11 dB) are as large as the effects being tested.
- `mix` changes two things: the recon trains on off-distribution masks for 25% of its steps, and the mask gets 25% fewer
  updates. Its curves were still rising at epoch 100, so 100 epochs may undercount `mix`; the frozen protocol scored epoch 100.
- The flat mask uses p = 0.5 before the budget rescale (p = 1 divides by zero in `RescaleProbMap`). After the rescale it is a
  uniform random mask at the same per-slice budget, as the proposal specified.
- The band rectangle and the 200 threshold come from the previous experiment's masks, chosen there after looking at them.
- wandb ran offline on CoreHPC. The offline run folders were copied with the runs but not synced.
- Checkpoints are untracked and live on `/export/scratch`.

## Figures

- `figures/masks.png`: learned p at epoch 100 for every arm and seed, with the right band outlined.
- `figures/curves.png`: trainer val PSNR over epochs (brain, knee), all 8 runs.
- `figures/recons_s43.png`: median-PSNR brain, knee-PD and knee-PDFS slices, seed 43. Shows the target, each arm's recon,
  |error| × 5 and the drawn mask.

## Next steps

- Test the hypothesis directly: start an lr 1e-3 run from an `lr4e3_step` VarNet at epoch 5 with a fresh mask, or swap
  VarNets between the two at epoch 5. This separates "better early VarNet" from "band present".
- If `mix` is pursued, run it to 150 epochs, or with a fraction that decays to 0 by epoch 30, so the recon ends on its own
  mask. Use 3+ seeds, since seed 42's knee-PD drop decides the arm.
- Use `lr4e3_step` (or a 6-cascade VarNet, now the default) as the LOUPE baseline. Both keep the band without any of these
  changes.

## Reproduce

`bash reports/loupe_mask_commitment_r8/run.sh` from the repo root with the env activated. It needs the 8 run folders under
`/export/scratch/bkadota/loupe_mask_commitment_r8/runs`, the previous experiment's `runs_full` lr 1e-3 controls, local v1 under
`/export/scratch/bkadota/fastmri_raw`, and one GPU (GPU 1). Check with
`python scripts/report_manifest.py check reports/loupe_mask_commitment_r8`.
