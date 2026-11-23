# Why does LOUPE drop the right k-space band, and which fix keeps it? (VarNet + LOUPE, R=8, 2026-11-19; follow-up 2026-11-23)

| | |
|---|---|
| Code | results produced at `17afaea` (loupe_mask_commitment_r8 follow-up: arm/seed options for the mask and recon scripts, follow-up criteria summary); branch `exp/loupe_training__loupe_mask_commitment_r8` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 20 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 21 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |
| Training commit | not recorded for these runs (they predate `run_info.json`) |


- **Branch**: `exp/loupe_training__loupe_mask_commitment_r8` (renamed from `exp/loupe_mask_commitment_r8` on 2026-11-20).
  Round-1 results at `287a06b`; follow-up results at `17afaea`; the branch tip after this report is cited on `main`.
- **Training commits.** Round 1: run state `0bf4cec` on CoreHPC (sbatch array 3118620; `run_info.json` has an empty commit
  because the CoreHPC code copy had no `.git`). Follow-up: `1bf63fd` (the 100-epoch runs, local GPUs 0–1) and `bc30363`
  (the 150-epoch runs, local GPUs 0–3), both recorded in each run's `run_info.json`.
- **Status: merged 2026-11-23** (human verdict). `sampler.flat_mask_fraction` (`mix`) and `sampler.flat_mask_decay_epochs`
  (`mixdecay`) go to `main` as options, off by default. The `clamp` (`loupe_logit_limit`) and `anneal` (`slope_start`) arms are
  parked: their code stays on the branch only.

### Proposal vs outcome, round 1 (criteria frozen in `proposal.md`, commit `75b14ca`; 100 epochs, seeds 42/43)

| arm | band kept: count >= 200, both seeds | fix works: mean gain >= +0.10 dB, both > 0, no condition < −0.05, SSIM >= −0.002 |
|---|---|---|
| `clamp` | **not met** (24.4 / 13.1) | **not met** (−0.020 / −0.061, mean −0.040) |
| `anneal` | **not met** (88.6 / 348.0) | **not met** (+0.005 / +0.065, mean +0.035) |
| `mix` | **met** (539.0 / 527.3) | **not met** (−0.018 / +0.095, mean +0.039; knee-PD s42 −0.122) |

Reading by the frozen rule: **inconclusive**.

### Proposal vs outcome, follow-up (criteria frozen in `followup_proposal.md`, commit `49279ab`; seeds 42/43/44)

| arm | vs | band kept: >= 200 in all seeds | fix works: mean >= +0.10 dB, all > 0, no condition < −0.05, SSIM >= −0.002 |
|---|---|---|---|
| `mixdecay` (100 ep) | 100-epoch control | **met** (485 / 470 / 297) | **not met** on the mean: +0.065 / +0.097 / +0.030, mean +0.064; worst condition +0.025, worst SSIM +0.0003 |
| `mix_e150` (150 ep) | `control_e150` | met in the 2 seeds run (535 / 516) | met in the 2 seeds run (+0.220 / +0.302, mean +0.261; worst condition +0.026, worst SSIM +0.0059) |

Reading by the frozen rule: **not evaluable**. The rule needs 3 seeds; the seed-44 `control_e150` / `mix_e150` pair was
still queued on CoreHPC when the experiment was closed and was cancelled. On the 2 seeds run, `mix_e150` would read "slow
convergence", but most of its margin is the 150-epoch control getting worse (Results, follow-up).

### Verdict and recommendation

The human verdict (2026-11-23): done. Keep `mix` and `mixdecay` on `main` as options; park everything else. The agent's
recommendation at writing agreed on the code: `mixdecay` is the only change that keeps the band in every seed and gains in
every seed and condition (+0.03 to +0.10 dB), though under the frozen +0.10 dB bar; `clamp` and `anneal` do not keep the band
reliably.

## Question

In `reports/varnet_loupe_lr_schedule_r8/report.md`, both 12-cascade lr 1e-3 controls left a band of k-space unsampled. The band
is rows 148–172 (within 12 of centre) and phase-encode columns 232 and up (more than 72 right of centre). The runs that kept
it were 0.12–0.17 dB better. This experiment asks whether the band is lost to sigmoid saturation, or to an early race in which a
young VarNet teaches the mask that the band is useless. It also asks whether either fix gains PSNR. "Band count" is the summed
LOUPE probability p = sigmoid(slope · logit) in that band, before the per-slice budget rescale (expected samples per slice).

The follow-up (`followup_proposal.md`) asks why `mix`, the only round-1 arm that kept the band, did not gain: (a) it converges
more slowly, so a 150-epoch schedule should lift it (`mix_e150` vs `control_e150`), or (b) its flat-mask batches tax the recon,
so stopping them early should give the gain at 100 epochs (`mixdecay`, flat fraction 0.25 → 0 over epochs 0–30).

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

**Follow-up runs** (same model, data and optimizer; seeds 42/43/44; trained on the local A6000s, data from
`/export/scratch/bkadota/fastmri_raw`, wandb online in groups `loupe_mask_commitment_r8_runs_e100` / `_runs_e150`):

| run | epochs, lr step | change vs control | band count | PSNR all | brain | knee-PD | knee-PDFS | SSIM all |
|---|---|---|---|---|---|---|---|---|
| control_s42 / s43 (round 1, CoreHPC) | 100, 90 | none | 5.8 / 2.9 | 33.820 / 33.799 | 32.967 / 32.940 | 36.001 / 36.002 | 32.377 / 32.340 | 0.8633 / 0.8621 |
| control_s44 | 100, 90 | none | 0.9 | 33.824 | 32.966 | 36.013 | 32.378 | 0.8630 |
| mixdecay_s42 | 100, 90 | flat fraction 0.25 → 0 over epochs 0–30 | 485.1 | 33.885 | 33.054 | 36.052 | 32.437 | 0.8638 |
| mixdecay_s43 | " | " | 469.8 | 33.896 | 33.058 | 36.066 | 32.450 | 0.8639 |
| mixdecay_s44 | " | " | 296.5 | 33.854 | 32.991 | 36.042 | 32.415 | 0.8634 |
| control_e150 s42 | 150, 135 | none | 0.0 | 33.690 | 32.737 | 36.017 | 32.191 | 0.8584 |
| control_e150 s43 | " | none | 4.6 | 33.655 | 32.742 | 35.892 | 32.211 | 0.8592 |
| mix_e150 s42 | 150, 135 | flat fraction 0.25, constant | 534.7 | 33.910 | 33.118 | 36.044 | 32.461 | 0.8650 |
| mix_e150 s43 | " | " | 516.0 | 33.957 | 33.165 | 36.104 | 32.494 | 0.8651 |

Checkpoints: `/export/scratch/bkadota/loupe_mask_commitment_r8/runs_e100/<run>` and `runs_e150/<run>` (symlinked as
`checkpoints/lmc_e100_<run>`, `checkpoints/lmc_e150_<run>`; `runs_e100/control_s42`/`s43` are links to round 1's runs).
Re-scoring round 1's controls in the follow-up eval gives their round-1 numbers exactly.

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
4. **Follow-up.** The same eval scores each follow-up folder (`--runs-dir runs_e100` / `runs_e150`) into
   `eval_followup_e100.json` / `eval_followup_e150.json`. The mask script scores them with `--arms/--seeds/--name`.
   `scripts/eval/loupe_mask_commitment_r8_followup_summary.py` applies the follow-up's frozen criteria per arm on the seeds
   present, and pairs 150-epoch runs against 100-epoch runs of the same seed on the same slices ("cross-length"). It also
   gives each run's best trainer val point against its last one.
5. **Recons.** `scripts/plots/loupe_mask_commitment_r8_recons.py` shows, per condition, the val slice at the median control
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

### Follow-up: paired PSNR (dB, full val)

| comparison, seed | all | 95% CI | brain | knee-PD | knee-PDFS | volumes better | SSIM diff |
|---|---|---|---|---|---|---|---|
| mixdecay − control (100 ep) s42 | +0.065 | [+0.043, +0.088] | +0.088 | +0.051 | +0.060 | 90% | +0.0005 |
| mixdecay − control (100 ep) s43 | +0.097 | [+0.075, +0.118] | +0.118 | +0.065 | +0.110 | 96% | +0.0018 |
| mixdecay − control (100 ep) s44 | +0.030 | [+0.012, +0.048] | +0.025 | +0.029 | +0.037 | 66% | +0.0003 |
| mix_e150 − control_e150 s42 | +0.220 | [+0.169, +0.272] | +0.381 | +0.026 | +0.270 | 91% | +0.0065 |
| mix_e150 − control_e150 s43 | +0.302 | [+0.265, +0.339] | +0.423 | +0.212 | +0.283 | 98% | +0.0059 |

`mixdecay` is positive in every seed and every condition, with every per-seed CI above zero, but its mean (+0.064 dB) is under
the +0.10 dB bar. Removing the flat batches after epoch 30 removed round 1's knee-PD loss (−0.122 in `mix_s42`).

### Follow-up: cross-length comparisons (same seed, same slices; dB)

| comparison | s42 (95% CI) | s43 (95% CI) | brain s42 / s43 |
|---|---|---|---|
| control_e150 − control (100 ep) | −0.130 [−0.171, −0.089] | −0.144 [−0.162, −0.126] | −0.230 / −0.199 |
| mix_e150 − control (100 ep) | +0.090 [+0.066, +0.115] | +0.158 [+0.129, +0.187] | +0.151 / +0.225 |
| mix_e150 − mixdecay (100 ep) | +0.025 [+0.010, +0.040] | +0.061 [+0.048, +0.075] | +0.064 / +0.107 |

The 150-epoch control is worse than the 100-epoch control, so about half of `mix_e150`'s +0.22 / +0.30 margin over its own
control is the control's decline. Against the better 100-epoch control, `mix_e150` still gains +0.09 / +0.16 dB.

### Follow-up: masks and curves

`mixdecay` keeps the band through 70 epochs without any flat batches (297–485, controls 0.9–5.8); its support (0.221–0.225)
and saturated share (0.749–0.756) match the control's. `mix_e150` keeps a band of 516–535 with the wider support of round-1
`mix` (0.230–0.232 vs 0.206–0.207 for `control_e150`). On trainer val (brain), both `control_e150` runs peak at epoch 70
(37.64 / 37.65 dB) and fall to 37.31 / 37.39 by epoch 150 while their training loss keeps falling (0.0218 vs 0.0225 at
100 epochs); `mix_e150` rises to the end (37.79 / 37.85) (`figures/curves_followup.png`). `mixdecay` ends above all three
100-epoch controls on brain from about epoch 65 (37.64–37.73 vs 37.60).

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

- **Follow-up: an early flat-mask phase is enough to keep the band.** `mixdecay` keeps it in all 3 seeds after 70 epochs of
  plain LOUPE training, so the band is lost early and, once kept, is not drained again. This supports the race reading.
- **Follow-up: keeping the band this way gives a small, consistent gain, under the frozen bar.** `mixdecay` gains +0.030 to
  +0.097 dB, positive in every seed, condition and per-seed CI, with no knee-PD cost. It does not reach +0.10 dB mean.
- **Follow-up: at 150 epochs the control overfits and `mix` does not.** Within-run trainer val shows `control_e150` falling
  ~0.3 dB on brain after epoch 70; the eval puts it 0.13–0.14 dB under the 100-epoch control. `mix_e150` beats its own control
  by +0.22 / +0.30 dB and the 100-epoch control by +0.09 / +0.16 dB. Hypothesis, untested: the flat-mask batches act as a
  regulariser on the recon, separately from keeping the band; these runs cannot split the two.
- **Open question raised at the verdict (hypothesis, untested).** Joint LOUPE + VarNet training may be stuck in a poor optimum
  because the young recon sends poor gradients to the mask early on, which is when the band is decided. Recon pretraining, or
  the flat-mask warm-up as a cheap form of it, is the natural test.

## Caveats

- Two seeds per arm. Seed-to-seed swings within an arm (0.04–0.11 dB) are as large as the effects being tested.
- `mix` changes two things: the recon trains on off-distribution masks for 25% of its steps, and the mask gets 25% fewer
  updates. Its curves were still rising at epoch 100, so 100 epochs may undercount `mix`; the frozen protocol scored epoch 100.
- The flat mask uses p = 0.5 before the budget rescale (p = 1 divides by zero in `RescaleProbMap`). After the rescale it is a
  uniform random mask at the same per-slice budget, as the proposal specified.
- The band rectangle and the 200 threshold come from the previous experiment's masks, chosen there after looking at them.
- wandb ran offline on CoreHPC. The offline run folders were copied with the runs but not synced.
- Checkpoints are untracked and live on `/export/scratch`.
- **Follow-up, missing seed.** The seed-44 `control_e150` / `mix_e150` pair was cancelled unrun when the experiment closed, so
  the 150-epoch arm has 2 of the 3 seeds the frozen rule needs.
- **Follow-up, mixed platforms.** The 100-epoch controls s42/s43 trained on CoreHPC (RTX Pro 6000); every other follow-up run
  trained locally (A6000). So `mixdecay` s42/s43 vs control and every cross-length comparison cross platforms; only
  `mixdecay_s44` vs `control_s44` is same-platform (+0.030 dB). The `control_e150` decline is also seen within each run's own
  trainer val curve, which no platform change touches.
- **Follow-up, schedule confound.** `control_e150` keeps lr 1e-3 until epoch 135; the 100-epoch control steps at 90. The decline
  may belong to the longer high-lr phase, not to the epoch count alone.

## Figures

- `figures/masks.png`: learned p at epoch 100 for every arm and seed, with the right band outlined.
- `figures/curves.png`: trainer val PSNR over epochs (brain, knee), all 8 runs.
- `figures/recons_s43.png`: median-PSNR brain, knee-PD and knee-PDFS slices, seed 43. Shows the target, each arm's recon,
  |error| × 5 and the drawn mask.
- `figures/masks_followup_e100.png`, `figures/masks_followup_e150.png`: learned p for the follow-up controls, `mixdecay` and
  `mix_e150`, band outlined.
- `figures/curves_followup.png`: trainer val PSNR of all 10 follow-up runs (100- and 150-epoch); shows the `control_e150` decline.
- `figures/recons_followup_e100_s43.png`, `figures/recons_followup_e150_s43.png`: control vs `mixdecay` (100 epochs) and
  control vs `mix_e150` (150 epochs) on the median-PSNR slices of seed 43, with error maps and mask draws.

## Next steps

Follow-up (2026-11-23), superseding the round-1 list below where they overlap:

- Test the verdict's open question: pretrain the VarNet (on random masks, or with the flat-mask warm-up) before the joint
  phase, and compare against `mixdecay` and the control at the same total epochs.
- Split band from regularisation: `mixdecay` against an early-stopped or step-at-epoch-70 control, and `mix` against a recon
  trained on random masks with the learned mask frozen.
- Repeat `mixdecay` at 6 cascades (the default since 2026-11-18) with 3 seeds on one platform.

Round 1:

- Test the hypothesis directly: start an lr 1e-3 run from an `lr4e3_step` VarNet at epoch 5 with a fresh mask, or swap
  VarNets between the two at epoch 5. This separates "better early VarNet" from "band present".
- If `mix` is pursued, run it to 150 epochs, or with a fraction that decays to 0 by epoch 30, so the recon ends on its own
  mask. Use 3+ seeds, since seed 42's knee-PD drop decides the arm.
- Use `lr4e3_step` (or a 6-cascade VarNet, now the default) as the LOUPE baseline. Both keep the band without any of these
  changes.

## Reproduce

`bash reports/loupe_mask_commitment_r8/run.sh` from the repo root with the env activated. It needs the 8 run folders under
`/export/scratch/bkadota/loupe_mask_commitment_r8/runs`, the previous experiment's `runs_full` lr 1e-3 controls, local v1 under
`/export/scratch/bkadota/fastmri_raw`, the follow-up folders `runs_e100` and `runs_e150`, and one GPU. Round 1's `clamp` and
`anneal` checkpoints need the parked code: run from the branch
(`git worktree add ../SeqMRI-worktrees/exp-loupe_training__loupe_mask_commitment_r8 exp/loupe_training__loupe_mask_commitment_r8`).
Check with
`python scripts/report_manifest.py check reports/loupe_mask_commitment_r8`.
