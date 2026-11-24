# Multi-step adaptive sampling — Roadmap

Generated from [design.md](design.md) on 2026-11-20. Edit freely; the loop runs the first unchecked item whose dependencies are done.

## Next

- [ ] **3. Train with exact 90° rotations and flips (D4 augmentation)** — branch not created yet
  - Question: does training on rotated and flipped scans make F's sampling density follow the scan, as SeqMRI's paper
    showed for rotated k-space, and does it change reconstruction quality? (design.md open question 4)
  - Arms: F_lrs, Fb_lrs and L with augmentation; controls are A's seed-42 runs. Proposal pending the user's OK.

- [ ] **2. Header-routed LOUPE against F on the 1.5 T scans** — branch not named yet
  - Question: is F's fscr1 lead scan-type routing? (design.md open question 2)
  - Not proposed yet; from the Next steps of `reports/multistep_adaptivity/report.md`.

## Later / unscheduled

- Multi-step sampling for SPACE, once the echo-train evolution is modelled.

## Done

- [x] **1. Sampler learning rate and per-step logit maps (proposal A)** — verdict: merge (2026-11-23)
  - Report: `reports/multistep_adaptivity__seqmri_lr_bias_r8/`
  - Branch: `exp/multistep_adaptivity__seqmri_lr_bias_r8` @ 661af17
  - Outcome: no frozen row passes; post hoc, F_lrs beats a LOUPE-level twin by +0.14 dB; densities barely follow a 90° rotation

- [x] **0. Order-free multi-step go/no-go** — verdict: NULL (2026-11-19)
  - Report: `reports/multistep_adaptivity/`
  - Branch: `exp/multistep_adaptivity` @ bca8acb (merged as main d498afd)
  - Design change: settled result added
