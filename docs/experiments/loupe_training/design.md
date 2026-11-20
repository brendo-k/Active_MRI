# LOUPE + VarNet training — Design

Status: active
Last verdict: 2026-11-17, parked (`reports/varnet_loupe_lr_schedule_r8/report.md`)

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

## Settled results (do not relitigate)

- **Recon lr × schedule** — PARKED: recon lr 4e-3 with a step at 90 is +0.12 to +0.17 dB over 1e-3. Evidence:
  `reports/varnet_loupe_lr_schedule_r8/report.md`. Reopen only if: the larger lr grid on more data is run (roadmap Later).

## Open questions

- Does the right k-space band that the lr 1e-3 controls drop explain their 0.12–0.17 dB deficit? — still open after roadmap item 1

## Success rule

Set per experiment in its `reports/<topic>/proposal.md`.

## Constraints

- None beyond `AGENTS.md`.

## Links

- Roadmap: [roadmap.md](roadmap.md)
- Reports: `reports/varnet_loupe_lr_schedule_r8/`; `reports/loupe_mask_commitment_r8/` (on its branch until a verdict)
- Related: `reports/varnet_train_speed/`, `reports/muon_layernorm_100ep/`, `reports/muon_layer_speed/` (VarNet training speed, merged)
