# Do Muon and LayerNorm still help 1-step LOUPE + VarNet (R=8) at full data and 100 epochs? (2026-11-05)

| | |
|---|---|
| Code | results produced at `fbbf1c8` (muon_layernorm_100ep summary: hours to target and each arm's own late gain); branch `exp/muon_layernorm_100ep` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 8 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 8 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |

- **Branch:** `exp/muon_layernorm_100ep`, not merged. Results come from the commit in the provenance block. All 8 runs trained
  from a detached snapshot at `91932da`, and nothing under `activemri/`, `configs/` or `train.py` changed after it on this branch.
- **Status:** waiting for the user's verdict. The proposal (`proposal.md`) was frozen at `4c5c2f3` on 2026-11-03, and the later
  amendments change the compute plan only (2 GPUs, the user's request).

## Proposal vs outcome

d(arm, seed) is the paired mean PSNR_Masked difference, arm − control, over all val slices (Methods 2).

| # | Criterion (frozen) | Evidence | Met? |
|---|---|---|---|
| A1 | Muon is better: mean of its 4 paired differences ≥ +0.15 dB, at least 3 positive | mean +0.216 dB; +0.124, +0.332, +0.171, +0.236; 4 of 4 positive | **met** |
| A2 | LayerNorm is better: the same rule | mean +0.120 dB; +0.014, +0.222, +0.091, +0.155; 4 of 4 positive | not met: the mean is under +0.15 dB |
| B | Muon + LayerNorm is better: d > 0 in both seeds, mean ≥ +0.15 dB | +0.346, +0.327; mean +0.336 dB | **met** |
| C | An arm is faster: reaches its control's final at ≤ 50% of the control's wall-clock, in both seeds | `muon` 40.9%, 40.0%; `muon_layer` 37.1%, 28.0%; `layer` 100.9%, 96.9% | **met** for `muon` and `muon_layer`; not met for `layer` |
| G1 | SSIM guard: paired difference ≥ −0.005 in both seeds | `muon` +0.0022, +0.0040; `layer` −0.0013, +0.0005; `muon_layer` +0.0046, +0.0045 | holds |
| G2 | Condition guard: no condition worse by ≥ 0.10 dB in both seeds | worst case `layer` brain −0.06, −0.02 dB | holds |
| G3 | Stability: no NaN loss, at most 10 consecutive skipped steps | 0 skipped steps and no non-finite epoch loss in all 8 runs | holds |
| S | Sanity: SSIM_Masked ≥ 0.80, control's final trainer PSNR ≥ 31.8 dB | lowest pooled SSIM 0.855; control 33.51, 33.28 dB | holds as frozen, on pooled SSIM; knee-PDFS alone is 0.78 in every arm (Caveats) |

Under the frozen decision rule, B is met with the guards holding, so the rule says to flip both defaults. LayerNorm alone is
**inconclusive**: A2 and C are not met, and neither is its failure condition.

The proposal's predictions held. Muon +0.1 to +0.3 dB came out +0.12 and +0.17. LayerNorm within ±0.1 dB came out +0.01 and +0.09.
Muon + LayerNorm +0.1 to +0.4 dB at ≤ 40% of the wall-clock came out +0.35 and +0.33 at 37% and 28%.

**Recommendation** (the user decides): **merge** this branch as a successful experiment.

This branch changes no default. The user set Muon + LayerNorm as the project default on 2026-11-04, ahead of this result, in
`553f760` on `exp/muon_layer_speed`, which is not merged. This result supports that change. The helper scripts both branches
share are identical, so the squash order does not matter for them.

- **Adopt the two together.** The combination is the only arm that clears the quality bar by a clear margin in both seeds.
- **Quote +0.34 dB, not +0.6 to +0.85 dB.** Those larger numbers came from short runs (Results 6). The direction holds and the
  size is about half.
- **A run that goes back to Adam should also go back to InstanceNorm.** LayerNorm with Adam gives +0.05 dB, costs 7% per epoch
  and lowers brain in both seeds (−0.06, −0.02 dB).
- **Muon without LayerNorm is the smaller effect**, +0.12 and +0.17 dB. This study tests Muon on this model only, so the default
  is untested on a model where `recon.norm` changes nothing.

## Question

At the real training budget (all of local v1 FLAIR + knee, 100 epochs, lr stepped at epoch 90), do Muon and LayerNorm still help
1-step LOUPE + E2E VarNet at R=8? There are two questions: is the 100-epoch model better, and does it reach the control's
100-epoch quality in less wall-clock? Earlier evidence came from short runs and was inconsistent: LayerNorm won on 20% of the data
and vanished at full data, while Muon did the reverse (`reports/varnet_train_speed/report.md`). The answer sets the training
recipe under every baseline on the roadmap, where effects of 0.05–0.2 dB are judged against a 0.3–0.5 dB seed spread.

Muon is orthogonalised momentum on the U-Nets' hidden convolutions, with Adam on everything else. LayerNorm is `GroupNorm(1, C)`
after every convolution of the cascade and sensitivity U-Nets. PSNR_Masked and SSIM_Masked are scored inside the apodized
per-anatomy metric mask.

## Runs compared

A 2×2 factorial with 2 seeds, 8 runs. Within a seed only the optimizer and the norm layer differ.

| Arm | Optimizer | Norm | Overrides beyond the common set | Runs (wandb) | PSNR_Masked / SSIM_Masked, s42; s43 |
|---|---|---|---|---|---|
| `control` | Adam | InstanceNorm | none | `control_s42` (o7t0yo37), `control_s43` (dr45hai8) | 33.51 / 0.859; 33.28 / 0.855 |
| `muon` | Muon | InstanceNorm | `training.optimizer=muon` | `muon_s42` (sfvpma2v), `muon_s43` (0ud9npcl) | 33.64 / 0.861; 33.45 / 0.859 |
| `layer` | Adam | LayerNorm | `recon.norm=layer sampler.sens_norm=layer` | `layer_s42` (uz6esl6k), `layer_s43` (5hjwc0e9) | 33.53 / 0.857; 33.37 / 0.856 |
| `muon_layer` | Muon | LayerNorm | both | `muon_layer_s42` (nidd1x7z), `muon_layer_s43` (467g3cyy) | 33.86 / 0.863; 33.61 / 0.860 |

Checkpoints are in `~/data/seqmri_checkpoints/muon_layernorm_100ep/<run>` (linked from `checkpoints/`). Every arm uses a 12-cascade,
18-channel VarNet, Adam or Muon at lr 1e-3, stepped by 0.1 at epoch 90, and `main`'s defaults at `91932da`: the fast path
and `sampler.sense_mask_method=null`. The common overrides are:

```
recon=varnet datasets=fastmri_local_v1_flair_knee misc.resolution=[320,320] sampler.accelerations=[8.0]
sampler.budget_over=acquired training.num_step=1 training.batch_size=16 training.num_epochs=100
training.lr_step_size=90 sampler.sampler_type=loupe_2d sampler.loupe_vd_init_power=4
sampler.anatomy_conditioning=false datasets.limit_volumes=1 training.seed={42,43}
```

The data are all of local v1 FLAIR + knee: 160 train volumes (3,929 slices) and 80 val volumes (1,938 slices: brain 605, knee-PD
673, knee-PDFS 660). The eval's PSNR matches the trainer's epoch-100 value to within 0.003 dB in every run.

## Methods

1. **Training.** The runs came from `scripts/launch/varnet_train_speed_worker.sh`, on GPUs 0–1, from 2026-11-03 12:23 to
   2026-11-05 07:21, in four waves of two: (`control_s42`, `muon_layer_s42`), (`layer_s42`, `muon_s42`),
   (`muon_layer_s43`, `control_s43`), (`layer_s43`, `muon_s43`). Each arm ran once on each GPU.
2. **Final quality.** `scripts/eval/varnet_train_speed_eval.py` scores each `last_model.pt` (epoch 100, so no selection on val) on
   all val slices, with 2 seeded mask draws (1234, 4321) and each slice's own data range. d(arm, seed) is the mean over slices of arm − control
   in the same seed. The 95% CI is ±1.96 standard errors over the per-volume mean differences, so it measures the spread across val
   volumes, not training noise.
3. **Factorial effects.** `scripts/eval/muon_layernorm_100ep_summary.py` computes each main effect as the mean of 4 paired differences:
   `muon − control` and `muon_layer − layer` for Muon, `layer − control` and `muon_layer − muon` for LayerNorm, each at both seeds.
   The interaction is `(muon_layer − layer) − (muon − control)` per seed.
4. **Time to target.** From the trainer's per-epoch val PSNR_Masked (`scripts/eval/varnet_train_speed_curves.py`, wandb and run
   logs). The target is the same-seed control's mean over epochs 96–100. The reach epoch is the first whose trailing 3-epoch
   mean meets it. The wall-clock share is reach epoch × the arm's median epoch time ÷ (100 × the control's median epoch time).
5. **Epoch times.** Medians over epochs 2–100 from the run logs, excluding epoch 1's compile.
6. **Late training (post-hoc, not in the proposal).** The same script reports window means of the trainer's val PSNR_Masked and each
   arm's lead over its control, in windows chosen after seeing the curves.

## Results

### 1. Final quality

| Arm | d, s42 [95% CI] | d, s43 [95% CI] | Mean | SSIM difference, s42, s43 | Volumes better of 80, s42, s43 |
|---|---|---|---|---|---|
| `muon` | +0.124 [+0.09, +0.16] | +0.171 [+0.11, +0.23] | +0.147 | +0.0022, +0.0040 | 68, 68 |
| `layer` | +0.014 [−0.01, +0.04] | +0.091 [+0.05, +0.13] | +0.052 | −0.0013, +0.0005 | 26, 50 |
| `muon_layer` | +0.346 [+0.28, +0.41] | +0.327 [+0.27, +0.38] | +0.336 | +0.0046, +0.0045 | 78, 75 |

The combination gains about 0.33 dB in both seeds, even though the control itself differs by 0.23 dB between seeds, and it
clears the 0.15 dB single-difference floor from the proposal in each. Volumes better counts the val volumes where the arm's mean
PSNR is higher than the control's.

### 2. Factorial effects

| Effect | s42 | s43 | Mean |
|---|---|---|---|
| Muon on InstanceNorm (`muon − control`) | +0.124 | +0.171 | +0.147 |
| Muon on LayerNorm (`muon_layer − layer`) | +0.332 | +0.236 | +0.284 |
| **Muon main effect** | | | **+0.216** |
| LayerNorm with Adam (`layer − control`) | +0.014 | +0.091 | +0.052 |
| LayerNorm with Muon (`muon_layer − muon`) | +0.222 | +0.155 | +0.189 |
| **LayerNorm main effect** | | | **+0.120** |
| Interaction | +0.209 | +0.065 | +0.137 |

Each factor adds more on top of the other, and all 8 differences are positive. The interaction has the predicted sign in both seeds but
is below the 0.15 dB floor in seed 43, so it is not established.

### 3. Per condition

| Arm | Brain, s42, s43 | Knee-PD, s42, s43 | Knee-PDFS, s42, s43 |
|---|---|---|---|
| `muon` | +0.13, +0.13 | +0.23, +0.28 | +0.01, +0.10 |
| `layer` | −0.06, −0.02 | +0.06, +0.17 | +0.03, +0.11 |
| `muon_layer` | +0.30, +0.20 | +0.63, +0.51 | +0.10, +0.25 |

Knee-PD gains the most in every arm. LayerNorm alone lowers brain slightly in both seeds, inside the guard; on seed 42 that is
−0.057 dB [−0.090, −0.025]. The +1.8 dB brain gain it showed on 20% of the data does not appear here.

### 4. Time to the control's final

| Arm | Reach epoch, s42, s43 | Wall-clock share, s42, s43 | C |
|---|---|---|---|
| `muon` | 39, 38 | 40.9%, 40.0% | met |
| `layer` | 94, 91 | 100.9%, 96.9% | not met |
| `muon_layer` | 33, 25 | 37.1%, 28.0% | met |

The targets are 33.512 dB and 33.275 dB. Muon + LayerNorm gets there in 3.7 and 2.8 h, against 10.0 h for 100 control epochs.
LayerNorm alone reaches the target only after the lr step.

### 5. Cost per epoch

| Arm (wave) | Train s, s42, s43 | Val s, s42, s43 | Epoch s, s42, s43 | Epoch vs control |
|---|---|---|---|---|
| `control` (1, 3) | 297.7, 298.5 | 60.8, 61.9 | 358.6, 361.4 | |
| `muon` (2, 4) | 315.4, 318.4 | 60.8, 61.3 | 376.5, 380.0 | +5.0%, +5.2% |
| `layer` (2, 4) | 321.5, 321.8 | 63.3, 62.9 | 385.0, 384.9 | +7.4%, +6.5% |
| `muon_layer` (1, 3) | 339.5, 342.2 | 63.3, 63.1 | 403.1, 405.4 | +12.4%, +12.2% |

Epoch medians of one arm differ by under 1% between its two waves. The two costs add up: training takes +17.7 and +19.9 s with Muon,
+23.8 and +23.3 s with LayerNorm, and +41.8 and +43.7 s with both.

`reports/varnet_train_speed/report.md` quoted +8% per epoch for the combination. That figure is low because the 12-epoch control
happened to validate slower (69.0 s, against 61.8–64.3 s for its arms). Its training-only overhead was +12.7%, close to the +14.1%
and +14.6% here.

### 6. Effect size against budget

Paired d in dB against the same-seed control. The first two rows are from `reports/varnet_train_speed/report.md` and used
`sampler.sense_mask_method=pre_network`; this study uses `null`.

| Budget | Muon | LayerNorm | Muon + LayerNorm |
|---|---|---|---|
| 20% of the data, 40 epochs | +0.10, −0.37 | +0.79, +0.88, +0.59 | +0.70, +1.00 |
| All data, 12 epochs | +0.26, +0.45 | +0.02, +0.05 | +0.60 (1 seed) |
| All data, 100 epochs (this study) | +0.12, +0.17 | +0.01, +0.09 | +0.35, +0.33 |

### 7. How the lead changes during training

Lead over the same-seed control, trainer val PSNR_Masked, mean over each window (dB):

| Arm | Seed | Epochs 46–50 | 66–70 | 86–90 | 96–100 | Own gain, 66–70 to 96–100 |
|---|---|---|---|---|---|---|
| `muon` | 42 | +0.46 | +0.35 | +0.20 | +0.13 | +0.01 |
| `muon` | 43 | +0.54 | +0.45 | +0.29 | +0.18 | +0.01 |
| `layer` | 42 | −0.15 | +0.02 | −0.05 | +0.01 | +0.23 |
| `layer` | 43 | +0.03 | +0.11 | +0.11 | +0.09 | +0.26 |
| `muon_layer` | 42 | +0.55 | +0.51 | +0.36 | +0.35 | +0.07 |
| `muon_layer` | 43 | +0.75 | +0.63 | +0.47 | +0.33 | −0.02 |
| `control` | 42 | | | | | +0.24 |
| `control` | 43 | | | | | +0.28 |

The leads fade because the Adam arms keep climbing, not because the Muon arms get worse. Both Muon arms are nearly flat from
epoch 70, while both Adam arms gain 0.23–0.28 dB. `muon_layer_s43` peaked at epoch 73 (33.65 dB) and ended 0.05 dB lower.
`control_s43` is still at its best at epoch 100.

## Conclusions

- **Muon + LayerNorm is better and faster at the real budget.** It gains +0.346 and +0.327 dB, with SSIM +0.0046 and +0.0045,
  and improves 78 and 75 of the 80 val volumes (Results 1). It reaches the control's final at 37% and 28% of the wall-clock, at
  +12% per epoch (Results 4, 5). All guards hold.
- **Muon helps by itself, by less.** A1 and C are met, but the +0.216 dB mean is carried by its LayerNorm pairs. On InstanceNorm it
  is +0.124 and +0.171 dB, under the 0.15 dB bar on average (Results 2).
- **LayerNorm helps only together with Muon.** With Adam it gives +0.014 and +0.091 dB, is not faster and costs 7% per epoch. With
  Muon it adds +0.222 and +0.155 dB. By the frozen rule it is inconclusive on its own and is adopted only through B.
- **The extra gain from combining them is not established.** The interaction is +0.209 and +0.065 dB, the second below the floor
  (Results 2).
- **Most of the gain is in knee-PD:** +0.63 and +0.51 dB, against +0.30 and +0.20 for brain and +0.10 and +0.25 for knee-PDFS
  (Results 3).
- **The final gain is about half of what short runs showed** (Results 6), though the change in sense mask is mixed into that comparison.
- **Hypothesis, untested:** the lead would shrink further with a longer schedule, because the control is still climbing at
  epoch 100 and the Muon arms are flat (Results 7). No run here goes past 100 epochs.
- **Hypothesis, untested:** Muon + LayerNorm needs fewer than 100 epochs, since it stops improving around epoch 70 in seed 43.

## Caveats

- **Two seeds.** The criteria need only two, and the differences are stable across them, but two seeds cannot estimate run-to-run
  noise. The proposal's same-seed noise, −0.27 ± 0.28 dB, was measured at 20% data and epochs 5–18, not on finals like these.
- **The CIs cover val volumes only**, not training noise, and the 4 differences of a main effect share seeds, so they are not
  independent.
- **Knee-PDFS SSIM is under the 0.80 line in every run.** S is frozen on pooled SSIM (0.855–0.863), but knee-PDFS alone is
  0.779–0.784 in all 8 runs, including the control. `AGENTS.md` treats SSIM below 0.80 as a bug. This predates the study:
  `fulldata_ref_s42` is 0.731 and the 12-epoch controls about 0.72 (`reports/varnet_train_speed/eval_vs_full.json`). The comparison is
  paired and no arm moves it by more than 0.005, so the verdict does not depend on it. It has not been debugged.
- **The +0.33 dB depends on the budget.** It is the gap at epoch 100 of this schedule, measured while the control is still improving.
- **Reach is measured inside a 100-epoch schedule**, with lr 1e-3 until epoch 90. A run stopped or stepped earlier is a different
  run. See Next steps 2.
- **`muon` and `layer` did not share a wave with their control**, as the proposal's amendment notes. `muon` at 40–41% would need a
  timing error of about 20–25% to miss the 50% bar, and an arm's medians move by under 1% between waves. `muon_layer` shares both
  waves with its control, and `layer` misses by far.
- **Other jobs shared the server.** An unrelated job ran on GPUs 2–3 during `layer_s42` and `muon_s42`. Their medians match the
  seed-43 runs to within 1%.
- **`control_s43` misses its own target** by 0.0002 dB on the trailing-3 rule, because it rises to the last epoch. This affects no arm.
- **The eval cannot split the gain between mask and networks.** Each arm learns its own mask, so a gain here is of the joint model.
- **Not comparable with older results.** `sampler.sense_mask_method=null` separates these runs from `fulldata_ref_s42` (32.11 dB) and
  the 12-epoch `full_*` runs.
- **Per-anatomy numbers.** Results 3 uses each slice's own data range. The `val/brain/PSNR` and `val/knee/PSNR` curves in wandb use the
  last batch's data range and do not average to the pooled number.
- **Post-hoc tables.** Results 6 and 7 and the per-condition SSIM values were not in the proposal.
- **Why LayerNorm costs more per epoch is not known.** `reports/muon_layer_speed/report.md`, on branch `exp/muon_layer_speed`, tried a
  LayerNorm that stays in bf16 and found it slower, so running in float32 is not the obvious cause.

## Figures

- `figures/curves.png`: val PSNR_Masked and SSIM_Masked against epoch and wall-clock, seed mean with the range shaded.
- `figures/lead_over_control.png`: each arm's lead over its same-seed control against epoch, one panel per seed. The colours differ
  from `curves.png`, so read the labels.

## Next steps

1. **Merge order.** Decide on this branch and `exp/muon_layer_speed` together, since that branch holds the default change this
   study supports.
2. **Retime `training.num_epochs`.** `e30_step_s42` in `reports/muon_layer_speed/report.md` (branch `exp/muon_layer_speed`) ended at
   33.55 dB in 30 epochs, close to this study's 100-epoch control at 33.51 dB on the same seed, but on different step code. Test
   30–70 epochs of the recipe with 2 seeds, including an lr step before epoch 90.
3. **Bound the budget dependence.** Extend one control and recipe pair past 100 epochs to see how much of the +0.33 dB survives.
4. **Look at knee-PDFS SSIM.** Find out why it sits at 0.78 while the other conditions are 0.88–0.92, and whether the 0.80 line should
   apply per condition.

## Reproduce

`bash reports/muon_layernorm_100ep/run.sh` regenerates every result file. It needs the 8 runs under
`~/data/seqmri_checkpoints/muon_layernorm_100ep`, one idle GPU and wandb access. The eval reuses `eval.per_slice.npz`: delete it to
rescore. Check the folder with `python scripts/report_manifest.py check reports/muon_layernorm_100ep`.

Scripts: `scripts/eval/varnet_train_speed_eval.py`, `scripts/eval/varnet_train_speed_curves.py`,
`scripts/eval/muon_layernorm_100ep_summary.py`, `scripts/plots/varnet_train_speed_curves.py`,
`scripts/plots/muon_layernorm_100ep_lead.py`.
