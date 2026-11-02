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

- [ ] **4. Field-strength masks (protocol × field)** — branch `exp/field_strength_masks_r8`
  - Question: Is field strength a header field worth routing on? Does a mask per field strength fix PDFS's loss without costing the other classes? (open question: field strength)
  - What we want out of it:
    - A yes/no on adding field strength to the routing table.
    - Whether field (SNR) changes the learned mask shape (1.5T more centre-heavy).
    - Whether halving the data per mask costs the classes that already gain (PD).
  - Control: the local v1 shared twins, seeds 42/43 (`b6ij26pe`, `mmd4sn0i`), reused through their stored per-slice scores. The 3-class runs are a second reference.
    - Code: the 10-26 snapshot, R over the 320² grid, VD p=4 start.
    - Gate before launch: restore the 10-26 code and the seed-42 shared checkpoint from the NFS snapshot (`~/data/.snapshot/Daily@2300.2026-10-27_2300/`, expires about 2026-11-05) to `/export/scratch`. Rescoring it must reproduce the stored scores. The current repo changed the output formation and the budget default since then.
  - Treatment: the same code, seeds, start and 100 epochs, trained from scratch, with 6 mask classes:
    - brain-1.5T, brain-3T, knee-PD-1.5T, knee-PD-3T, knee-PDFS-1.5T, knee-PDFS-3T.
    - Field comes from the manifest's `field` column, which was read from the scan header.
  - Expected outcome:
    - 3T PDFS rises from −0.055 / −0.033 to ≥ 0, its mask spreading outward toward PD's (mean radius ≈ 0.50). 1.5T PDFS stays ≈ 0. PDFS overall reaches ≥ +0.02.
    - PD keeps +0.07 / +0.12 and brain overall is unchanged.
    - Brain-1.5T is noisy: 19 train / 10 val volumes.
  - Success criterion (all three in both seeds):
    1. PDFS (20 volumes) ≥ +0.02 dB over the shared twin, and above the 3-class run's PDFS gap.
    2. PD within 0.03 dB of the 3-class PD gain.
    3. Brain overall within 0.03 dB of the 3-class brain gain.
    - Failure: criterion 1 fails in both seeds, or 2 fails in either. Seeds that disagree make it inconclusive.
    - Secondary readouts (not criteria): per-field gain for each protocol, and each mask's mean radius.
  - Failure means:
    - 3T PDFS still losing with its own mask → the 3T deficit is the shared reconstructor, not the mask (item 6).
    - PD dropping more than 0.03 → splitting the data costs more than field-specific masks gain; field routing is NO-GO at this data size.
    - Either way, add the verdict under settled results.
  - Caveats:
    - Field is confounded with scanner model, so a field mask is also a scanner mask.
    - With 10 val volumes per group, per-group gaps under about 0.05 dB are not resolved.
    - 2 seeds is below the design's 3-seed rule. A third seed (44) needs a new shared twin as well: +2 runs, about 12.5 h.
  - Depends on: none, but the snapshot restore must happen before about 2026-11-05.
  - Compute: about 30 min for the reproducibility gate, then 2 runs on GPUs 0–1 in parallel (about 12.5 h) and about 30 min of eval per seed. Outputs on `/export/scratch`.

- [ ] **5. FLAIR density control** — branch `exp/flair_density_r8`
  - Question: Is the local v1 FLAIR gain about brain, or about sampling FLAIR at R≈5.2 of its acquired k-space? (open question: FLAIR density)
  - Control: shared twin with `sampler.budget_over=acquired`, which puts FLAIR at R=8 of acquired k-space like knee. Current code.
  - Treatment: the 3-class per-protocol masks, same settings.
  - Expected outcome: if density drives the gain, the FLAIR gain shrinks toward the T2 result (≤ 0) while PD is unchanged.
  - Success criterion: FLAIR per-condition minus shared CI excludes 0 in both seeds. If so, the FLAIR mask gain is about the anatomy.
  - Failure means: the FLAIR gain came from the denser sampling. Record it under settled results and stop treating FLAIR as evidence for brain routing.
  - Depends on: none
  - Compute: 4 runs (2 seeds × 2 arms), about 25 h on GPUs 0–1.

- [ ] **6. Reconstructor conditioning with class-specific masks** — branch `exp/recon_conditioning_masks_r8`
  - Question: Does one blind VarNet lose by serving class-specific masks, and does a class label (FiLM in every cascade) recover it? (open question: conditioned reconstructor)
  - Control: masks frozen from a trained pair, blind VarNet trained from scratch. The design is a 2×2 of {shared mask, per-class masks} × {blind, FiLM}.
  - Treatment: FiLM-conditioned VarNet on the same frozen masks. The test is the interaction: conditioning should help more with per-class masks than with the shared mask.
  - Expected outcome: small; the mask and image already identify the class. A gain under per-class masks only would mean interference between classes in the network.
  - Success criterion: (FiLM − blind) with per-class masks exceeds (FiLM − blind) with the shared mask, with a CI excluding 0.
  - Failure means: the shared network is not the bottleneck; drop label conditioning of the reconstructor.
  - Notes:
    - On FLAIR + knee with grid R, the shared mask already differs by class (FLAIR denser and cut to a narrower region). Use `budget_over=acquired` (after item 5) or T2 + knee, so the shared row really is one mask distribution.
    - Freezing a 3-mask checkpoint needs a small change to `fixed_mask_checkpoints`.
    - Seed-43 masks were lost in the 10-28 cleanup.
  - Depends on: 5 (decides the budget setting)
  - Compute: 4 runs, about 25 h on GPUs 0–1.

## Later / unscheduled

- Replicate on the internal SPACE data once the anatomy pair arrives and IRB is confirmed.
- Complex-image or k-space loss (user interest; RSS L1 leaves the maps' common phase free).
- Sens U-Net bilinear upsample vs LOUPE edge energy (inconclusive 15-epoch run).

## Done

- [x] **0a. Anatomy pair, random start (ISMRM 2027)** — verdict: PARTIAL (2026-10-21)
  - Report: `reports/ismrm2027/`, `reports/anatcond_r8/`
  - Branch: n/a (not run through the loop)
  - Design change: knee mask is knee-specific; brain gain not mask-specific.
- [x] **0b. VD-start and same-start anatomy pairs** — verdict: MIXED (2026-10-24/25)
  - Report: `reports/ismrm2027/vdinit_summary.json`
  - Branch: n/a
  - Design change: Decision 2 and the "knee gain depends on the start" result.
- [x] **0d. Local v1 per-protocol masks (FLAIR / PD / PDFS)** — verdict: small PASS for FLAIR and PD, NULL for PDFS (2026-10-28)
  - Report: `reports/local_v1_flair_knee/anatomy_pair_vdinit_p4_seed{42,43}.json`
  - Branch: n/a (run before the loop; checkpoints deleted 10-28, per-slice scores kept)
  - Design change: the per-protocol and 3T PDFS entries under settled results; items 4–6.
- [x] **0c. Patient-specific conv sampler (local v1)** — verdict: NO-GO for patient-specific (2026-10-28)
  - Report: `reports/patient_specific_r8_local_v1/seed42/`
  - Branch: n/a
  - Design change: Decision 4 and the conv sampler result.
