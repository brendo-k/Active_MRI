# Anatomy / header routing — Design

Status: active 
(anatomy / header routing)

## Question

When training a brain, knee pd, knee pdfs seperate masking for LOUPE, we get worse performance on the brain compared to a single mask for all three. This is when training using the VarNet. However, when using CG-SENSE, we get better performance using anatomy specific. Can we evaluate why? 

## Fixed setup

- Dataset: local v1, FLAIR + knee-PD + knee-PDFS, 320² grid, manifest loader (`fastmri_local_v1_flair_knee`). Train on 0.5 percent of the data
- Acceleration: R=8. 320x320 grid, all anatomies
- Reconstructor: E2E VarNet, 12 cascades, bs 16, 100 epochs, lr 1e-3 stepped at epoch 90. Sense net for sensetivity maps.
- Sampler: `loupe_2d` straight-through. Shared and per-anatomy rows start from the same draw.
- Metric: trainer's apodized per-anatomy metric mask. Healthy SSIM about 0.88–0.93; below 0.80 means a bug. Never compare numbers across those dates or use eval scripts that score full images.

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
- Is the limited performance due to network co-adaption? Is it because the network now has to learn two sampling patterns for two anatomies? 


## Success rule

We figure on the current best masking strategies.

## Constraints

- No compute on the HPC login node. Crashed runs are rerun in full (no resume).
- Checkpoints live in `~/data/seqmri_checkpoints`; optimizer state is stripped.
- Brain val outlier AXT2_201_2010571 (~25 dB): pick figure slices by PSNR rank.
- Single-pass adaptivity is about 0 against a non-adaptive twin in the literature (`docs/LITERATURE_ADAPTIVE_SAMPLING_R8.md`), so expect small effects.

## Links

- Roadmap: [roadmap.md](roadmap.md)
- Related docs: `docs/idea.md`, `docs/ADAPTIVE_MASK_EXPERIMENT_PLAN.md`, `docs/ISMRM_2027_ABSTRACT.md`, `docs/ACS_ROUTER.md`
- Reports: `reports/ismrm2027/`, `reports/anatcond_r8/`, `reports/cgsense_anatcond/`, `reports/local_v1_flair_knee/`, `reports/patient_specific_r8_local_v1/`, `reports/varnet_caipi_promptmr_anatcond_r4/`
