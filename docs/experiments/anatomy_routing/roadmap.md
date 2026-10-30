# Anatomy / header routing — Roadmap

## Next

- [ ] **1. More seeds on the VD-start anatomy pair** — branch `exp/anatomy_pair_seeds`
  - Question: Is the knee routing gain (+0.05 / +0.13 dB at p=4) real beyond seed spread? (open question 1)
  - Control: shared LOUPE twin, true pair, VD p=4, same code and metric.
  - Treatment: per-anatomy rows, same VD p=4 start, seeds 44–46 added to 42/43.
  - Expected outcome: knee +0.05 to +0.15 dB; brain 0 or negative.
  - Success criterion: knee delta CI excludes 0 over 5 seeds with at least 4 of 5 positive, and no significant brain loss.
  - Failure means: knee routing is within noise; add a NULL entry to design.md and stop spending on routing by label.
  - Depends on: none
  - Compute: 6 runs × about 100 epochs at about 171 s/epoch (≈ 5 h each), 4 GPUs.

- [ ] **2. Coil-count header lookup for brain** — branch `exp/coil_count_routing`
  - Question: Does routing brain by coil-count class (few-coil vs many-coil) beat the single brain mask? (open question 2)
  - Control: shared twin and brain/knee anatomy routing, same VD p=4 start.
  - Treatment: three classes (few-coil brain, many-coil brain, knee), header lookup only. Fix the 16-coil truncation first, or run both pipelines identically.
  - Expected outcome: +0.02 to +0.1 dB on few-coil brain; the effect is small and few-coil volumes are few.
  - Success criterion: few-coil brain delta CI excludes 0 on held-out volumes (not the 23 used to pick the route), no regression on many-coil brain.
  - Failure means: add a NO-GO for coil-count routing and drop the idea in `docs/idea.md`.
  - Depends on: 1 (reuse its pairing and start setup)
  - Compute: about 6 runs; check the count of few-coil volumes in train/val before running. Likely too few; if so, fix the split first.

- [ ] **3. Different reconstructor control (PromptMR backbone)** — branch `exp/anatomy_pair_promptmr`
  - Question: Does the routed knee gain survive a stronger reconstructor? (open question 3)
  - Control: shared twin with `backbone=promptmr`, lr 2e-4.
  - Treatment: per-anatomy rows, VD p=4, same backbone.
  - Expected outcome: gain equal or smaller than item 1.
  - Success criterion: same rule as item 1 on 2 seeds; sign matches item 1.
  - Failure means: the gain is specific to the plain VarNet; limit claims to it.
  - Depends on: 1
  - Compute: 4 runs; PromptMR is about 83M params (slower than the 30M VarNet).

## Later / unscheduled

- Replicate on the internal SPACE data once the anatomy pair arrives and IRB is confirmed.
- Complex-image or k-space loss (user interest; RSS L1 leaves the maps' common phase free).
- Sens U-Net bilinear upsample vs LOUPE edge energy (inconclusive 15-epoch run).
- Local v1 FLAIR/PD/PDFS 3-class vs shared run (`reports/local_v1_flair_knee/`): needs write-up into the design.md results table once evaluated.

## Done

- [x] **0a. Anatomy pair, random start (ISMRM 2027)** — verdict: PARTIAL (2026-10-21)
  - Report: `reports/ismrm2027/`, `reports/anatcond_r8/`
  - Branch: n/a (not run through the loop)
  - Design change: knee mask is knee-specific; brain gain not mask-specific.
- [x] **0b. VD-start and same-start anatomy pairs** — verdict: MIXED (2026-10-24/25)
  - Report: `reports/ismrm2027/vdinit_summary.json`
  - Branch: n/a
  - Design change: Decision 2 and the "knee gain depends on the start" result.
- [x] **0c. Patient-specific conv sampler (local v1)** — verdict: NO-GO for patient-specific (2026-10-28)
  - Report: `reports/patient_specific_r8_local_v1/seed42/`
  - Branch: n/a
  - Design change: Decision 4 and the conv sampler result.
