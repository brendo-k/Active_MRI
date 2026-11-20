# LOUPE + VarNet training — Roadmap

Generated from [design.md](design.md) on 2026-11-20. Edit freely; the loop runs the first unchecked item whose dependencies are done.

## Awaiting verdict

- [ ] **1. Mask commitment** — branch `exp/loupe_training__loupe_mask_commitment_r8`
  - Renamed from `exp/loupe_mask_commitment_r8` on 2026-11-20; its report still names the old branch.
  - Question: why does LOUPE drop the right k-space band, and which fix keeps it: a logit clamp, a slope anneal, or
    batches with a flat mask?
  - Proposal and report: `reports/loupe_mask_commitment_r8/`, on the branch
  - Result: inconclusive by the frozen rule. No arm passes both criteria, and saturation is ruled out as the cause. The
    recommendation at writing is park.

## Later / unscheduled

- A larger recon lr grid (8e-3, 1e-2 and beyond) on more data (`reports/varnet_loupe_lr_schedule_r8/report.md`, Next steps 1).

## Done

- [x] **0. Recon lr × schedule** — verdict: parked (2026-11-17)
  - Report: `reports/varnet_loupe_lr_schedule_r8/`
  - Branch: `exp/varnet_loupe_lr_schedule_r8` @ 2e6e37f (parked; report on main f550c15)
  - Design change: decision 1
