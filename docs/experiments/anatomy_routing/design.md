# Anatomy / header routing — Design

Status: active 
(anatomy / header routing)

## Question

Can a mask chosen before the scan from header information (anatomy, protocol, coil count, field strength) beat one shared learned mask at R=8, and by more than the seed spread? Routing by header is the only adaptive lever so far with a consistent positive sign (knee), so it comes before ACS budget adaptation and multi-step sampling.

## Fixed setup

- Dataset: ISMRM 2027 pair uses `brain_knee_balanced` (brain / knee-PD / knee-PDFS). Local v1 uses FLAIR + knee on a 320² grid via the manifest loader (default since 10-27).
- Acceleration: R=8. Since 10-27 the budget is of the acquired k-space (`sampler.budget_over=acquired`), so FLAIR and local-v1 numbers are not comparable to the grid-R pairs. The local v1 3-class pair (started 10-26) predates that change and counts R over the 320² grid. FLAIR acquires 65% of the grid (40–80%), so it was sampled at R≈5.2 of its acquired k-space, against 8 for knee.
- Reconstructor: E2E VarNet, 12 cascades, bs 16, 100 epochs, lr 1e-3 stepped at epoch 90. BART ecalib sense maps (tuned crop + soft).
- Sampler: `loupe_2d` straight-through. Shared and per-anatomy rows start from the same draw (VD start `loupe_vd_init_power=4`, or `loupe_same_start`).
- Metric: trainer's apodized per-anatomy metric mask (changed 2026-10-20; `SSIM_Masked` fixed 2026-10-23). Healthy SSIM about 0.88–0.93; below 0.80 means a bug. Never compare numbers across those dates or use eval scripts that score full images.
- Controls: shared-mask twin with identical reconstructor/sense init and batch order (true pair), label-swap cost (other anatomy's mask on the same recon), and a condition-mean mask for the patient-specific question.
- Noise floor: 0.3–0.5 dB seed spread of joint LOUPE+VarNet training. Routing effects are 0.05–0.2 dB. Use true pairs, two or more seeds, and report run-minus-twin deltas, not absolute PSNR.

## Decisions

| # | Decision | Why | Evidence |
|---|----------|-----|----------|
| 1 | Route by header fields, not by an ACS-learned router | The mask must be fixed before the scan, and header fields are known by then | `docs/idea.md`, `docs/ACS_ROUTER.md` |
| 2 | Start per-class rows from the same draw, with a VD start | Random-start per-class rows collapsed (brain worse in every true pair) | `reports/ismrm2027/anatomy_pair_*`, `vdinit_summary.json` |
| 3 | Compare as true pairs (same recon/sense init, same batch order) | The original ISMRM pairs were not true pairs; the extra mask draw shifted init and batch order | pairing fix in the vdinit runs |
| 4 | Judge adaptivity by own minus condition-mean mask, not by swap cost | Swap cost mostly measures the penalty of another volume's deviation | `reports/patient_specific_r8_local_v1/` |
| 5 | BART ecalib sense maps, no post-hoc smoothing | Ties the net in brain, +1.6 dB in knee | `reports/pisco_sense_maps/` |

## Settled results (do not relitigate)

- **Knee mask is knee-specific** — PASS (VarNet R=8): swapping in the brain mask costs knee 0.13–0.29 dB in every run. Brain with the knee mask is -0.03 to +0.05. Evidence: `reports/ismrm2027/`. Reopen only if: the metric or reconstructor changes.
- **Brain gains nothing from a brain-specific mask** — NO-GO (VarNet R=8): across 6 true pairs routed brain is never positive (-0.02 to -0.69); the shared mask is the best brain mask. The old ISMRM s43 brain +0.13 was reconstructor co-adaptation. Evidence: `reports/ismrm2027/vdinit_summary.json`. Reopen only if: new brain data or a new mask family.
- **Knee gain depends on the start** — MIXED: knee gains in 3 of 6 true pairs, and only the VD p=4 start is positive in both seeds (+0.05 / +0.13 dB). Same-start seed 43 collapsed (-0.26). Not a density fix: final r≥64 share is about 0.50–0.54 from every start. Evidence: `reports/ismrm2027/anatomy_pair_vdinit_p*`, `anatomy_pair_samestart_p0_*`. Reopen only if: more seeds or a different optimizer for the per-class rows.
- **CG-SENSE per-anatomy knee deficit was mostly the start** — hybrid: from the shared mask the knee deficit shrinks from -0.25 to -0.035 dB, with a residual PDFS -0.06. Evidence: `reports/cgsense_anatcond/`. Reopen only if: the CG-SENSE straight-through gradient is fixed.
- **Label does not help a prompt-based VarNet** — NULL: cond minus blind ±0.08 dB over 2 seeds at CAIPI R=4, though a wrong label costs 0.1–0.8 dB. Evidence: `reports/varnet_caipi_promptmr_anatcond_r4/report.md`.
- **Patient-specific conv sampler** — NO-GO for the patient-specific claim: swap cost passes (knee-PD +0.049, PDFS +0.050 dB) but own minus condition-mean is about 0 (+0.012 PD, +0.005 PDFS, -0.034 brain). Conv trails LOUPE by 0.20 dB. Masks correlate 0.98 across volumes. Evidence: `reports/patient_specific_r8_local_v1/seed42/`. Seed 42 only.
- **Per-protocol masks on local v1 (FLAIR / PD / PDFS)** — small PASS for FLAIR and PD, NULL for PDFS. Setup: VarNet R=8 over the 320² grid, 2 seeds, true pairs.
  - Per-condition minus shared twin: FLAIR +0.07 / +0.12, PD +0.07 / +0.12 dB (19 of 20 volumes positive in each seed), PDFS −0.02 / −0.015.
  - Not mask under-training. Each class mask is updated on almost every step, since a batch of 16 mixes the classes. Class masks moved as far from the VD start as the shared mask (42–57% vs 49% of the budget), with the same Adam step sizes. The twin gap is flat from epoch 5.
  - The masks are clearly different (29–50% of the budget placed differently) yet end within about 0.1 dB of each other. Large swap costs (0.2–0.56 dB) mostly measure recon co-adaptation.
  - The per-condition run has one shared VarNet.
  - Evidence: `reports/local_v1_flair_knee/anatomy_pair_vdinit_p4_seed{42,43}.json`. Both runs' checkpoints were deleted 10-28; the per-slice scores remain in the `.pt` files.
  - Reopen only if: the budget rule or the reconstructor changes.
- **PDFS loss sits in the 3T volumes** — OBSERVATION, per volume and 2 seeds, 10 val volumes per field.
  - 3T PDFS −0.055 / −0.033 dB (0/10 and 2/10 volumes positive). 1.5T PDFS +0.01 / +0.01. PD gains at both fields.
  - The PDFS mask is the most centre-heavy (mean radius 0.45, vs PD 0.50 and shared 0.48).
  - On the per-condition recon, swapping in the PD mask costs 1.5T PDFS 0.07–0.11 dB but gives 3T PDFS +0.01–0.04. Even with the PD mask, 3T PDFS only ties the shared run.
  - Field is confounded with scanner model: knee 1.5T is all Aera; 3T is Skyra, Prisma and Biograph.
  - Evidence: per-slice scores in `reports/local_v1_flair_knee/*.pt`, joined to the manifest's `field` column. Reopen only if: more val volumes per field.
- **Coil count predicts brain routing** — SUGGESTIVE only: per-volume correlation of knee-vs-brain mask preference with coil count is 0.78 (field strength 0.59, base PSNR 0.27); 5 few-coil volumes, confounded with Avanto/TrioTim, routes picked on the same 23 validation volumes. Size about +0.02 dB on brain. Evidence: `docs/idea.md`. Not tested as a trained arm.

## Open questions

- Is the knee gain real beyond seed noise? Only 2 seeds so far, and seed 43 collapsed under two of three starts.
- Does coil count (header lookup) route brain better than the brain/knee label? Untested as a trained arm. Truncating to 16 coils may confound it.
- Does a routed mask survive a different reconstructor (PromptMR backbone, lr 2e-4)?
- Why do routed rows co-adapt the reconstructor (old s43 brain +0.13), and can that be separated from a mask gain?
- Does field strength (SNR) need its own mask? One PDFS mask fits noisy 1.5T PDFS, and 3T PDFS loses with it. Roadmap item 4.
- Is the local v1 FLAIR gain about brain, or about FLAIR's zero-filled k-space? With R over the grid, FLAIR is sampled at R≈5.2 of what it acquires. The fully acquired T2 pairs (R=8) never gained. Roadmap item 5.
- Does a class-conditioned reconstructor help once the masks differ by class? The earlier NULL had one CAIPI mask for every class, so the input distribution was not bimodal. Roadmap item 6.
- Does any of this hold on the in-house SPACE data (anatomy pair due 10-23)? IRB status unconfirmed.

## Success rule

Pre-registered for ISMRM 2027: per-anatomy beats shared with a 95% CI excluding 0 on at least one anatomy and no significant loss on the other, in both seeds. That failed on 2026-10-21 (brain never positive). For new work, restate per experiment before running: true-pair delta vs shared twin, at least 3 seeds, CI over volumes, and a seed-swap control. The target is a routed-minus-shared gain above 0.1 dB with consistent sign across seeds.

## Constraints

- No compute on the HPC login node. Crashed runs are rerun in full (no resume).
- Checkpoints live in `~/data/seqmri_checkpoints`; optimizer state is stripped.
- Brain val outlier AXT2_201_2010571 (~25 dB): pick figure slices by PSNR rank.
- Pre-10-27 conv/adaptive checkpoints saw a different sampler input (conj bug).
- Single-pass adaptivity is about 0 against a non-adaptive twin in the literature (`docs/LITERATURE_ADAPTIVE_SAMPLING_R8.md`), so expect small effects.

## Links

- Roadmap: [roadmap.md](roadmap.md)
- Related docs: `docs/idea.md`, `docs/ADAPTIVE_MASK_EXPERIMENT_PLAN.md`, `docs/ISMRM_2027_ABSTRACT.md`, `docs/ACS_ROUTER.md`
- Reports: `reports/ismrm2027/`, `reports/anatcond_r8/`, `reports/cgsense_anatcond/`, `reports/local_v1_flair_knee/`, `reports/patient_specific_r8_local_v1/`, `reports/varnet_caipi_promptmr_anatcond_r4/`
