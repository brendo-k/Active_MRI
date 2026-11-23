# LOUPE + VarNet training — Design

Status: active
Last verdict: 2026-11-23, merged: mix options on `main`, clamp and anneal parked (`reports/loupe_mask_commitment_r8/report.md`)

## Question

Which training setup gives the best joint LOUPE + VarNet at R=8: the recon learning rate and schedule, and how the LOUPE
mask commits to its samples? And why do some runs leave a band of k-space unsampled?

## Fixed setup

- Dataset: full local v1 FLAIR + knee (160 training volumes)
- Acceleration: R=8, `sampler.budget_over=acquired`
- Reconstructor: E2E VarNet, 12 cascades × 18 channels, Muon + LayerNorm; batch 16, 100 epochs
- Sampler: LOUPE `loupe_2d` from a uniform-noise start, slope 10
- Metric path: trainer apodized metric mask; PSNR / NMSE / SSIM_Masked
- Controls: recon lr 1e-3 with a step at epoch 90 (the default on `main`)
- Single seed is fine for now

## Decisions

| # | Decision | Why | Evidence |
|---|----------|-----|----------|
| 1 | `main` keeps `training.lr=1e-3` | lr 4e-3 gains +0.12 to +0.17 dB, but the grid stops while the gain still rises, and the data is 160 volumes | `reports/varnet_loupe_lr_schedule_r8/report.md` (user verdict 2026-11-17) |
| 2 | `main` gains `sampler.flat_mask_fraction` (`mix`) and `sampler.flat_mask_decay_epochs` (`mixdecay`), off by default; the user keeps `mix` in use for now | The only changes that keep the right band; `mixdecay` gains +0.03 to +0.10 dB in 3/3 seeds, `mix` stops the 150-epoch overfit | `reports/loupe_mask_commitment_r8/report.md` (user verdict 2026-11-23) |

## Settled results (do not relitigate)

- **Mask commitment** — MERGED (options only): sigmoid saturation is not why the band is lost (`clamp`); a flat-mask phase
  in the first 30 epochs keeps it for good (`mixdecay`, 3/3 seeds) and gains +0.064 dB mean, under the +0.10 bar; the
  12-cascade lr 1e-3 control overfits by 150 epochs (−0.13 dB), constant `mix` does not. Evidence:
  `reports/loupe_mask_commitment_r8/report.md`. Clamp and anneal code parked on the branch.
- **Recon lr × schedule** — PARKED: recon lr 4e-3 with a step at 90 is +0.12 to +0.17 dB over 1e-3. Evidence:
  `reports/varnet_loupe_lr_schedule_r8/report.md`. Reopen only if: the larger lr grid on more data is run (roadmap Later).

## Open questions

- Does the right k-space band that the lr 1e-3 controls drop explain their 0.12–0.17 dB deficit? — partly: keeping it is worth
  about +0.06 dB (`mixdecay`); the rest is open
- (User, 2026-11-23; hypothesis, untested) Joint LOUPE + VarNet training settles in a poor optimum because the young
  recon sends poor gradients to the mask early, when the band is decided. Test: recon pretraining before the joint phase,
  against `mixdecay` and the control at equal total epochs
- Do the flat-mask batches help as a regulariser on the recon, separately from keeping the band? (`mix_e150` vs an
  early-stopped control)

## Success rule

Set per experiment in its `reports/<topic>/proposal.md`.

## Constraints

- None beyond `AGENTS.md`.

## Links

- Roadmap: [roadmap.md](roadmap.md)
- Reports: `reports/varnet_loupe_lr_schedule_r8/`; `reports/loupe_mask_commitment_r8/`
- Related: `reports/varnet_train_speed/`, `reports/muon_layernorm_100ep/`, `reports/muon_layer_speed/` (VarNet training speed, merged)
