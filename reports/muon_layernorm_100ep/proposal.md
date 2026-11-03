# Proposal: Muon x LayerNorm at the real budget (full data, 100 epochs), 1-step LOUPE + VarNet R=8 (2026-11-03)

- **Topic / branch**: `muon_layernorm_100ep`, branch `exp/muon_layernorm_100ep`, report `reports/muon_layernorm_100ep/`.
- **Requested by**: the user, 2026-11-03: "run the full-data 100-epoch LayerNorm x Muon 2x2". This is Next step 2 of
  `reports/varnet_train_speed/report.md`. The design below follows that step; it was frozen without a separate approval
  round, so the user can still veto it (`touch ~/data/seqmri_checkpoints/muon_layernorm_100ep/STOP`).

## Question

At the real training budget (all of local v1 FLAIR + knee, 100 epochs, the lr stepped at epoch 90), do Muon and
LayerNorm still help 1-step LOUPE + E2E VarNet at R=8? There are two questions:
- **Better**: is the 100-epoch model better?
- **Faster**: does it reach the control's 100-epoch quality in less wall-clock?

The answer decides whether either becomes `main`'s default.

The two treatments:
- **Muon**: orthogonalised momentum on the U-Nets' hidden convolutions, Adam on everything else.
- **LayerNorm**: `GroupNorm(1, C)` after every convolution of the cascade and sensitivity U-Nets.

Why it matters for the roadmap: anatomy routing (step 1) and budget adaptation (step 2) are judged on 0.05-0.2 dB
effects against a 0.3-0.5 dB seed spread.
- A recipe that is 0.4-0.6 dB better moves every baseline's reference point, so the baselines would need retraining.
- A recipe that reaches the same quality in a fraction of the time buys more seeds per GPU-day.

Evidence so far, from `reports/varnet_train_speed`. All of it was measured with `sampler.sense_mask_method=pre_network`.
- 20% data, 40 epochs, paired eval vs the same-seed control:
  - LayerNorm +0.79 / +0.88 / +0.59 dB (3 seeds; brain +1.8);
  - Muon +0.10 / -0.37;
  - Muon + LayerNorm +0.70 / +1.00.
- Full data, 12 epochs (lr step at 11):
  - LayerNorm +0.02 / +0.05;
  - Muon +0.26 / +0.45;
  - Muon + LayerNorm +0.60 (1 seed). It reached the control's epoch-12 value at epoch 3.

## Control

`main` at the launch commit, with its defaults:
- Adam at lr 1e-3, stepped by 0.1 at epoch 90;
- InstanceNorm in the cascade and sensitivity U-Nets;
- the fast path (`train_metrics_every=10`, `visualize_every=10`, `recon.fused_coil_ops=true`);
- `sampler.sense_mask_method=null`, the default since 6bf37c0.

Every arm gets these overrides:

```
recon=varnet datasets=fastmri_local_v1_flair_knee misc.resolution=[320,320] sampler.accelerations=[8.0]
sampler.budget_over=acquired training.num_step=1 training.batch_size=16 training.num_epochs=100
training.lr_step_size=90 sampler.sampler_type=loupe_2d sampler.loupe_vd_init_power=4
sampler.anatomy_conditioning=false datasets.limit_volumes=1 training.seed={42,43}
```

The data are all of local v1 FLAIR + knee-PD + knee-PDFS (val 1,938 slices). The metric is PSNR_Masked / SSIM_Masked
inside the apodized per-anatomy metric mask.

No existing run is reused as the control; the ones that exist are not comparable with this study:
- The 100-epoch reference `fulldata_ref_s42` (loss_domain rss v1 at 1ba1ccf, 32.11 dB) uses older code and
  `pre_network`.
- The 12-epoch `full_*` runs use `pre_network`.
- In `reports/sense_mask_varnet`, `null` beat `pre_network` by 1.26 dB (1 seed, 50% data, 50 epochs).

## Experiment

A 2x2 factorial with 4 arms, 2 seeds each (42, 43), 8 runs:

| Arm | Optimizer | Norm (cascades and sensitivity net) | Overrides beyond the common set |
|---|---|---|---|
| `control` | Adam | InstanceNorm | none |
| `muon` | Muon | InstanceNorm | `training.optimizer=muon` |
| `layer` | Adam | LayerNorm | `recon.norm=layer sampler.sens_norm=layer` |
| `muon_layer` | Muon | LayerNorm | both |

Within a seed, only the optimizer and the norm layer differ. The seed, data, mask initialisation, budget, schedule
and code are the same.

## Measurements

- **Final quality.** Use `scripts/eval/varnet_train_speed_eval.py --queue-dir <queue>` on each run's
  `last_model.pt` (epoch 100, so no selection on val).
  - It scores the full val set with per-slice data range and 2 seeded mask draws (1234, 4321).
  - The paired difference is arm - control in the same seed, per slice. Report the mean, the 95% CI over volumes,
    and brain / knee-PD / knee-PDFS separately.
- **Time to target.** Use the trainer's per-epoch val PSNR_Masked (full val set, every epoch).
  - The target is the seed-matched control's mean over epochs 96-100.
  - The reach epoch is the first epoch whose trailing 3-epoch mean is at or above the target.
  - Wall-clock is the reach epoch times the arm's median epoch time (train plus val), as a fraction of 100 times the
    control's median epoch time.
  - The arms of one seed train at the same time on separate GPUs, so their epoch times are comparable.
- **Factorial effects.** Each main effect uses 4 paired differences:
  - Muon: `muon - control` and `muon_layer - layer`, each at s42 and s43.
  - LayerNorm: `layer - control` and `muon_layer - muon`, each at s42 and s43.
  - Interaction: `(muon_layer - layer) - (muon - control)`, per seed.

## Noise floor

- Same-seed run-to-run noise: `control_short_s42 - control_s42`, identical configs at 20% data over epochs 5-18, is
  -0.27 +/- 0.28 dB.
- Seed spread at full data, 12 epochs: 0.33 dB (control s42 vs s43).
- The paired per-slice eval with fixed mask draws removes the mask-draw noise, but not the training noise. A single
  difference below ~0.15 dB is therefore not evidence; a main effect needs 4 differences.

## Expected outcomes

- **Control**: trainer full-val PSNR_Masked at epoch 100 of about 32.5-33.5 dB. That is the old 32.11 reference plus
  part of the `null` sense-mask gain. SSIM_Masked >= 0.82.
- **Muon**: final +0.1 to +0.3 dB. The 12-epoch lead of +0.26 / +0.45 shrinks as both arms converge. It reaches the
  control's final in <= 50% of the control's wall-clock.
- **LayerNorm**: final within +/-0.1 dB of the control, as at 12 epochs on full data. Its 20%-data gain was a
  generalisation effect, mostly in brain. Brain is also where `null` gained most (+2.2 dB) over `pre_network`, so
  part of that gain may have been compensating for the hard support mask. With `null` it may vanish.
- **Muon + LayerNorm**: final +0.1 to +0.4 dB, and it reaches the control's final in <= 40% of the wall-clock. The
  interaction is positive (the combination beats the sum of the single effects), as at 12 epochs.
- **What the alternatives mean**:
  - All arms within +/-0.1 dB at 100 epochs, but the treatments reach that level sooner: a speed win. Adopt, and
    retime `num_epochs` as a follow-up.
  - Muon's final gain >= 0.15 dB: Muon finds better solutions, not just faster ones.
  - Muon worse at 100 epochs: keep Adam; Muon helps only short budgets.
  - LayerNorm worse with `null`: its earlier gain depended on `pre_network`.

## Success criteria (frozen)

Let d(arm, seed) be the paired mean PSNR_Masked difference, arm - control, over all val slices from the eval script.

| # | Criterion | Passes when |
|---|---|---|
| A1 | Muon is better (main effect) | the mean of its 4 paired differences is >= +0.15 dB, and >= 3 of the 4 are positive |
| A2 | LayerNorm is better (main effect) | the same rule, on its 4 differences |
| B | Muon + LayerNorm is better | d(muon_layer, s) > 0 in both seeds, and the mean is >= +0.15 dB |
| C | An arm is faster | it reaches its seed-matched control's final at <= 50% of the control's wall-clock, in both seeds (judged separately for `muon`, `layer`, `muon_layer`) |
| G1 | Guard: SSIM | the arm's paired SSIM_Masked difference is >= -0.005 in both seeds |
| G2 | Guard: per condition | no condition (brain / knee-PD / knee-PDFS) is worse by >= 0.10 dB in both seeds |
| G3 | Guard: stability | no NaN loss; no run of more than 10 consecutive skipped optimizer steps |
| S | Sanity (any run) | SSIM_Masked >= 0.80, and the control's final trainer val PSNR_Masked >= 31.8 dB; otherwise debug before reporting |

Decision rule. This is a recommendation; the user decides.
- **Quality decides first.** Recommend flipping a factor's default if:
  - its A criterion passes with G1-G3 holding: `training.optimizer=muon` for Muon, or
    `recon.norm=layer sampler.sens_norm=layer` for LayerNorm;
  - or, for both factors together, B passes with the guards holding.
- **Speed alone.** If an arm passes C, its final mean difference is >= -0.05 dB and the guards hold, recommend the flip
  as a speed win. Say plainly that the "better" criterion failed, and propose retiming `num_epochs`.
- **Failure** for a factor: the mean of its 4 differences is <= -0.15 dB with >= 3 of the 4 negative, or a guard fails
  in both seeds.
- **Inconclusive**: any other outcome.

## Compute plan

- **Runs**: 8 runs of 100 epochs, one per A6000, in two waves.
  - Wave 1 is the 4 arms at s42: a full 1-seed 2x2, for an interim look only; no decision is made from it.
  - Wave 2 is the 4 arms at s43.
- **Time**: about 375-405 s per epoch, so 10.5-11.5 h per run and about 23 h in total. The ETA is around midday
  2026-11-04 (PST).
- **Queue**: `~/data/seqmri_checkpoints/muon_layernorm_100ep/` (`jobs.txt`, `claims/`, logs).
  - The workers run a copy of `scripts/launch/varnet_train_speed_worker.sh`, parameterised for this queue.
  - The code comes from a detached snapshot worktree at the launch commit.
  - wandb: group `muon_layernorm_100ep`, runs named `ml100_<arm>_s<seed>`.
- **Crashes**: a crashed run is rerun from scratch, up to 3 attempts.
- **Eval**: after both waves, about 10 min per run on a free GPU.
