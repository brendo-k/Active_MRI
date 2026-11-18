# Proposal: lr and lr schedule for joint VarNet + LOUPE at R=8 (local v1, 20% data, 400 epochs, CoreHPC)

**Status: approved by the user 2026-11-12 ("Keep the shared arm, stash the config edit, go ahead").**

- Topic / branch: `varnet_loupe_lr_schedule_r8`, branch `exp/varnet_loupe_lr_schedule_r8`,
  worktree `../SeqMRI-worktrees/exp-varnet_loupe_lr_schedule_r8`, report folder `reports/varnet_loupe_lr_schedule_r8/`.
- Roadmap: infrastructure for every step. The effects we chase are 0.05-0.2 dB, so an under-tuned optimizer is a cheap
  place to lose separation.

## Question

Is lr 1e-3 with a step near the end still the best lr and schedule for joint VarNet + LOUPE at R=8 with today's defaults
(Muon + LayerNorm, local v1, brain-flip fix, exact step code)?

The last sweep (`reports/lr_sweep_r8`, 2026-10-19) does not answer it: Adam, a frozen VD mask, old data and old code.
Muon has since changed what an lr means (orthogonalised updates, RMS-matched to Adam), `training.muon_lr=2e-3` was only
tried at 30 epochs and mixed with a code change (`reports/muon_layer_speed`), and `brain_static_vs_loupe` found no run
converged by 65 epochs.

## Regime

- Data: `datasets.limit_volumes=0.2` of local v1 FLAIR + knee (fixed draw seed 1234, about 16 volumes per condition).
- 400 epochs, batch 16. Exact steps per epoch are measured in the smoke test and recorded in the report; the plan
  assumes about 50, so about 20k steps (full-data 100-epoch runs are about 25k).
- Step boundaries scale with the run: the step sits at 90% of training (epoch 360 of 400, like 90 of 100).
- Warmup stays at about 500 steps: `training.lr_warmup_epochs=10` (2 epochs at full data).
- `limit_volumes` also cuts val. Training validates every 10 epochs on the cut val, for curves only. The decision is
  scored afterwards on the **full** local v1 val (`limit_volumes=1`) from `last_model.pt` through the trainer's metric
  path (apodized metric mask).
- Every run, control included, runs on CoreHPC with the same code snapshot and the same `dl` env. Local runs are not mixed in.

## Control

`train.py --model sequential recon=varnet datasets=fastmri_local_v1_flair_knee misc.resolution=[320,320]
sampler.accelerations=[8.0] sampler.budget_over=acquired training.num_step=1 training.batch_size=16
training.num_epochs=400 sampler.sampler_type=loupe_2d sampler.anatomy_conditioning=false sampler.loupe_vd_init_power=0
datasets.limit_volumes=0.2 training.val_every=10`, main defaults otherwise (Muon + LayerNorm, `sense_mask_method=null`).
lr 1e-3 (recon) and `sampler.lr=0.001` (mask logits), step at epoch 360, gamma 0.1. Seeds 42, 43, 44. Retrained: no
existing run matches this data size, code or environment.

## Experiment

The mask-logit lr is pinned at `sampler.lr=0.001` in every arm except `shared_2e3_step`. Adam moves each logit about lr
per step, so the mask lr sets how far the mask can travel; changing `training.lr` alone would otherwise change the
recon and the mask search together. The schedule multiplies every parameter group, the mask included (LambdaLR / StepLR
scale each group's base lr), so a schedule arm changes the mask's schedule too; that is part of the treatment.

Grid: recon lr {5e-4, 1e-3, 2e-3} x schedule {step, warmup_cosine, wsd}, plus two single arms.

| arm | `training.lr` | `sampler.lr` | schedule |
|---|---|---|---|
| `lr1e3_step` (**control**) | 1e-3 | 1e-3 | step at 360, x0.1 |
| `lr5e4_step` | 5e-4 | 1e-3 | step at 360 |
| `lr2e3_step` | 2e-3 | 1e-3 | step at 360 |
| `lr5e4_wcos` | 5e-4 | 1e-3 | warmup_cosine, warmup 10 ep, floor 1% |
| `lr1e3_wcos` | 1e-3 | 1e-3 | warmup_cosine |
| `lr2e3_wcos` | 2e-3 | 1e-3 | warmup_cosine |
| `lr5e4_wsd` | 5e-4 | 1e-3 | wsd: warmup 10 ep, constant, linear decay to 0 over the last 20% (epoch 320 on) |
| `lr1e3_wsd` | 1e-3 | 1e-3 | wsd |
| `lr2e3_wsd` | 2e-3 | 1e-3 | wsd |
| `lr1e3_const` | 1e-3 | 1e-3 | constant (what decay buys at all) |
| `shared_2e3_step` | 2e-3 | 2e-3 | step at 360 (twin of `lr2e3_step`: does the mask lr matter?) |

`wsd` (warmup-stable-decay) is new code: `training.lr_scheduler=wsd`, `training.lr_decay_fraction=0.2`. The Muon group
follows `training.lr` (`muon_lr=null`), the Adam rest of the recon too.

## Noise floor

Seed spread of final PSNR is about 0.23-0.25 dB but shared across arms: in `muon_layernorm_100ep` the paired same-seed
difference repeated within 0.02 dB (+0.344 / +0.327). Training is not bit-deterministic (first-epoch loss differs in the
4th digit), and small data may widen seed effects (`brain_static_vs_loupe`: 0.44 dB static seed spread at 25% brain).
Every comparison is paired per seed and per slice on the full val set.

## Expected outcomes

- Step vs warmup_cosine at 1e-3: tie within 0.1 dB. Cosine lost at 1e-3 on Adam (-0.26 dB per slice), but spent less
  time at high lr; at 400 epochs it has more.
- wsd: ties or beats step by up to +0.1 dB (longer high-lr phase than cosine, longer decay than step).
- lr 2e-3: 0 to +0.15 dB over 1e-3 at matched schedule, if anything; Muon may be under-stepped at 1e-3.
- lr 5e-4: -0.1 dB or worse with step; closer with cosine/wsd.
- constant: -0.2 to -0.5 dB vs control (the step at 90 has been worth about that historically).
- `shared_2e3_step` vs `lr2e3_step`: within 0.05 dB if the mask lr does not matter; a larger difference makes mask
  lr its own experiment.
- SSIM moves with PSNR; NMSE opposite. Knee-PDFS SSIM is about 0.78 in current runs (below the 0.80 line, predates this
  study) and is reported, not debugged here.
- Masks: with the mask lr pinned, budget relocated between arms should be within seed noise. Larger shifts mean the
  recon lr steers the mask search.

## Success criteria (frozen)

Primary metric: paired PSNR_Masked difference vs the control, same seed, mean over all full-val slices.

- **Adopt** an arm when all hold: mean over seeds >= +0.10 dB; positive in 3/3 seeds; no condition (brain, knee-PD,
  knee-PDFS) worse than -0.05 dB on the seed mean; SSIM_Masked seed mean not worse than -0.002. If several pass, the
  largest mean wins. Picking the best of 10 arms biases the winner up; the 3/3 rule is the guard.
- **Inconclusive**: the best arm gains 0 to +0.10 dB, or is positive in only 2/3 seeds. Control stays the default.
- **Control confirmed**: every arm's seed mean <= 0.
- **Mask-lr question** (reported, not a default change): `shared_2e3_step - lr2e3_step`, |mean| >= 0.10 dB in 3/3
  seeds same sign = mask lr matters.
- Transfer: an adopted winner is a 20%-data / 400-epoch result. Flipping main's default needs a full-data confirmation
  on its own branch.

## Compute plan

- CoreHPC `gpu` partition (L40S / RTX Pro 6000 / H100 NVL), account `hpc_core`, 1 GPU per run, sbatch array of
  33 tasks (11 arms x 3 seeds). Estimated 5-8 h per run, well under the 4-day limit. No compute on the login node.
- Code ships as a snapshot of the branch run-state commit to `~/scratch/varnet_loupe_lr_schedule_r8/code`. Data from
  `~/scratch/data/fastmri/{anatomy}/data` via `datasets.roots.hpc` / `datasets.roots.local_batch2`; the manifest's
  tier-S files are checked to exist there first.
- Smoke test first (sbatch, 2 arms, few epochs): config resolves as intended, steps per epoch, memory, epoch time,
  0 skipped steps.
- wandb offline on CoreHPC, group `varnet_loupe_lr_schedule_r8`, runs `vlls_<arm>_s<seed>`; synced afterwards.
- Eval on CoreHPC too (same env), full val. Crashed runs rerun in full.
