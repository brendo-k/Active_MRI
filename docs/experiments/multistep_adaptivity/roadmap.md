# Multi-step adaptive sampling — Roadmap

Generated from [design.md](design.md) on 2026-11-20. Edit freely; the loop runs the first unchecked item whose dependencies are done.

## In flight

- [ ] **1. Sampler learning rate and per-step logit maps (proposal A)** — branch `exp/multistep_adaptivity__seqmri_lr_bias_r8`
  - Renamed from `exp/seqmri_lr_bias_r8` on 2026-11-20, with its report, launch and CoreHPC folders.
  - Question: does F gain from 50 more epochs at lr 1e-4, or from a sampler lr of 1e-4, and do per-step logit maps
    help under a route that credits every step? (design.md open question 1)
  - Proposal: `reports/multistep_adaptivity__seqmri_lr_bias_r8/proposal.md` and `addendum_2026-11-19_route.md`, on the branch
  - Control: fscr1's seed-42 tier-F runs (F, Fb, L)
  - Status: the 2026-11-19 CoreHPC validations failed on two bugs on `main` (fixed in 86b9a22). Resubmitted 2026-11-20
    at run state 3 (49be699): validations 3161832 (resume and sampler-lr arms) and 3161838 (route arms), then training
    3161833–3161837 and 3161839–3161842.

## Next

- [ ] **2. Header-routed LOUPE against F on the 1.5 T scans** — branch not named yet
  - Question: is F's fscr1 lead scan-type routing? (design.md open question 2)
  - Not proposed yet; from the Next steps of `reports/multistep_adaptivity/report.md`.

## Later / unscheduled

- Multi-step sampling for SPACE, once the echo-train evolution is modelled.

## Done

- [x] **0. Order-free multi-step go/no-go** — verdict: NULL (2026-11-19)
  - Report: `reports/multistep_adaptivity/`
  - Branch: `exp/multistep_adaptivity` @ bca8acb (merged as main d498afd)
  - Design change: settled result added
