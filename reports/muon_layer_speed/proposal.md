# Proposal: muon_layer_speed

**Status:** frozen 2026-11-03 23:00. The user approved it in chat with "Run on all the gpus until 8am". The criteria below
are not edited after this commit.

- **Topic / branch:** `muon_layer_speed`, branch `exp/muon_layer_speed`, report `reports/muon_layer_speed/`.
- **Question:** how much less wall-clock does 1-step LOUPE + 12-cascade E2E VarNet (R=8, full local v1 FLAIR + knee) with
  Muon + LayerNorm need to reach its 100-epoch quality? There are two parts:
  1. **Step time:** exact changes to the forward and backward pass.
  2. **Steps to converge:** shorter schedules, a higher Muon lr and a smaller batch.

  Why it matters: roadmap effects are 0.05-0.2 dB against 0.3-0.5 dB seed spread, so every question needs several seeds,
  and a 100-epoch run takes 11 h.
- **Baseline assumption (user, 2026-11-03):** Muon + LayerNorm counts as true, without waiting for the
  muon_layernorm_100ep 2x2.

## Control
- The 100-epoch run settings of `~/data/seqmri_checkpoints/muon_layernorm_100ep/jobs.txt` plus
  `training.optimizer=muon recon.norm=layer sampler.sens_norm=layer`, on main's fast path.
- The reference is `ml100_muon_layer_s42` (wandb `nidd1x7z`) and `ml100_muon_layer_s43`. The control is not retrained.
- At the freeze, `ml100_muon_layer_s42` is at epoch 91: val PSNR_Masked 33.86 dB, SSIM_Masked 0.863, 1.38 s/step, 402 s/epoch
  (339 s train, 63 s val). `ml100_control_s42` (Adam + InstanceNorm) finished at 33.51 / 0.859, 1.21 s/step.

## Part 1: step time
Profile the control on an idle GPU first; every profile on file is Adam + InstanceNorm. Then, ranked by that profile:
- LayerNorm: its cost is in training only. On 20% data its validation pass is faster than InstanceNorm's (14.4 vs 15.3 s)
  while a training step is 11.5% slower. Candidate: the same math through a cheaper reduction.
- Padded coils: 24.9% of the 20 coil slots hold zeros (FLAIR + knee train). The sensitivity net skips them, the cascades do not.
- Unused forward work for LOUPE: the sampler's input image, the zero-filled recon whose loss is only logged, and the
  k-space prediction after the last step.
- The Muon step: a Python loop per parameter; `_foreach_` ops and compiled Newton-Schulz iterations.
- Channels-last with LayerNorm, never benchmarked.

Not redone: sensitivity-net resolution, sensitivity estimator swaps, lite U-Net, coil chunking, exhaustive cuDNN search,
channels-last with InstanceNorm.

## Part 2: steps to converge
Full data; 20% data misled the last study (LayerNorm's gain vanished at full data, Muon's sign flipped). Every arm is the
control with one change, seeds 42 and 43.

| arm | change | epochs | share of reference wall-clock |
|---|---|---|---|
| `e50_step` | lr step at 45 | 50 | 50% |
| `e30_step` | lr step at 27 | 30 | 30% |
| `e30_cos` | `training.lr_scheduler=cosine` | 30 | 30% |
| `e30_cos_muon2e3` | cosine and `training.muon_lr=0.002` | 30 | 30% |
| `e30_bs8` | `training.batch_size=8`, same lr, lr step at 27 | 30 | about 33% |

Every arm also keeps an EMA of the reconstructor and sensitivity-net weights (decay 0.999), which leaves the training
trajectory unchanged and is scored at the end beside the last weights. The LOUPE mask stays at its online value.

## Noise floor
Two seeds, with paired per-slice scores on the checkpoint against the same-seed reference. Single epochs are not used: the
reference dropped 0.19 dB at epoch 90 and recovered at epoch 91.

## Metrics
`varnet_train_speed_eval.py`: PSNR_Masked and SSIM_Masked inside the apodized per-anatomy metric mask, 2 seeded mask draws,
per condition (brain, knee-PD, knee-PDFS), 95% CIs over per-volume means. Step time from an idle-GPU benchmark.

## Expected outcomes
- Step time: LayerNorm rewrite -5 to -8%, padded coils -5 to -8%, unused work about -2%, Muon up to -4%, channels-last
  probably little. These are estimates until the profile.
- Delta PSNR against the reference: `e50_step` about -0.1 dB, `e30_step` about -0.2 dB, `e30_cos` about -0.1 dB.
  `e30_cos_muon2e3` and `e30_bs8` are unknown; either could beat `e30_cos`, and the higher lr could be unstable.
- EMA: +0.03 to +0.1 dB over the last weights.

## Success criteria
- **Exact step-time change: adopt** if outputs and gradients match to float rounding (bitwise, or at most 1e-5 relative
  where only reduction order changes) and the benchmark step or epoch time drops by at least 5%. A bitwise-identical
  removal of unused work is adopted at any gain.
- **Convergence arm: faster** if its mean paired Delta PSNR against the reference is at least -0.10 dB, neither seed is
  below -0.20 dB, Delta SSIM is at least -0.002, and its wall-clock is at most 50% of the reference's.
- **Worse:** mean Delta PSNR below -0.25 dB.
- **Inconclusive:** anything else.
- **EMA: adopt** if it beats the last weights by at least 0.05 dB in both seeds of an arm.

## Compute plan
- The user set the limit at 08:00 on 2026-11-04. Runs must end by 07:15 to leave time for the evals and the report.
- The muon_layernorm_100ep 2x2 holds GPUs 0-1, and stopping it was blocked at the freeze, so this study starts on GPUs 2-3. If
  GPUs 0-1 are released, their workers join the same queue.
- Arm runs use `training.val_every=5`, because a 30-epoch run then takes about 2.9 h instead of 3.4 h.
- Jobs run from `~/data/seqmri_checkpoints/muon_layer_speed/`, in this order: seed 42 of `e30_cos`, `e30_step`, `e30_bs8`,
  `e30_cos_muon2e3`, `e50_step`, then seed 43 in the same order. A run that stops early is rerun from scratch, up to 3
  attempts.
- On two GPUs only seed 42 of about 4 arms can finish by 07:15, and `ml100_muon_layer_s43` will not exist by then. Any verdict
  at 08:00 from one seed is provisional: a criterion that needs two seeds is reported as not yet decided.
