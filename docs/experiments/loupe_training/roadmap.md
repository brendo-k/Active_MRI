# LOUPE + VarNet training — Roadmap

Generated from [design.md](design.md) on 2026-11-20. Edit freely; the loop runs the first unchecked item whose dependencies are done.

## Awaiting verdict

(none)

## Next

- [ ] **2. Recon pretraining before joint LOUPE training** (user's hypothesis, 2026-11-23): does a VarNet pretrained on
  random masks, then trained jointly with LOUPE, beat `mixdecay` and the control at the same total epochs? 6 cascades,
  3 seeds, one platform.
- [ ] **3. Band vs regularisation**: `mixdecay` / `mix` against an early-stopped (or step-at-70) control, to split the band
  effect from the overfitting `mix` prevents.

## Later / unscheduled

- A larger recon lr grid (8e-3, 1e-2 and beyond) on more data (`reports/varnet_loupe_lr_schedule_r8/report.md`, Next steps 1).

## Done

- [x] **1. Mask commitment** — verdict: merged, options only (2026-11-23)
  - Report: `reports/loupe_mask_commitment_r8/`
  - Branch: `exp/loupe_training__loupe_mask_commitment_r8` @ b25a8d1
  - Result: clamp does not keep the band; `mixdecay` keeps it in 3/3 seeds, +0.064 dB (bar +0.10); control overfits at 150
    epochs and `mix` does not. `mix` and `mixdecay` are options on `main`; clamp and anneal parked on the branch.
  - Design change: decision 2

- [x] **0. Recon lr × schedule** — verdict: parked (2026-11-17)
  - Report: `reports/varnet_loupe_lr_schedule_r8/`
  - Branch: `exp/varnet_loupe_lr_schedule_r8` @ 2e6e37f (parked; report on main f550c15)
  - Design change: decision 1
