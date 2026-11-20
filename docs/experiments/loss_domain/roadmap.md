# Loss domain — Roadmap

Generated from [design.md](design.md) on 2026-10-30. Edit freely; the loop runs the first unchecked item whose dependencies are done.

## Cache

Built 2026-10-29. `scripts/data/build_complex_image_cache.py` serves each slice through `FastMRIDataset` (readout crop, 320 grid, 20 coils, center-RSS norm), runs that anatomy's preset, coil-combines, and stores the image. 480 volumes, 10,270 slices that pass the loader's energy fraction of 0.2, 5.1 GB at `/export/scratch/bkadota/fastmri_complex_espirit`. Maps are not in the files. Eight recomputed slices matched to 0.

`M` is recomputed from each saved image and is not stored: 1 where `|x| > 0`, else 0. Mask-off arms leave it at 1. Before item 1, report how `M` overlaps the apodized metric mask (open question 4).

## Arms

RSS image L1 is the baseline on every item. It is the production loss (`loss_type=l1`): L1 of the coil-combined RSS magnitude, no `M`. The loss bar is that arm minus its RSS-L1 twin. The mask bar is masked minus unmasked inside one domain. Score stays RSS PSNR / NMSE / SSIM_Masked. Every L1 below is divided by the per-example L1 norm of its own target.

| Arm | Domain | Loss | Mask | Item |
|-----|--------|------|------|------|
| RSS L1 | image | production RSS magnitude | off | baseline, items 1–3 |
| Complex L1 | image | `recon_complex` vs cached `x` | off | 1, control |
| Complex L1 | image | `M · recon_complex` vs `M · x` | on | 1, treatment |
| SENSE k-space L1 | k-space | `FFT(S · image)` | off | 2, control |
| SENSE k-space L1 | k-space | `FFT(S · M · image)` | on | 2, treatment |
| RSS k-space L1 | k-space | `FFT(rss)` of the production RSS images | off | 3, control |
| RSS k-space L1 | k-space | `FFT(M · rss)` | on | 3, treatment |

Shared on every arm: LOUPE, VD p=4, VarNet R=8, `budget_over=acquired`, `coil_combine=sense`, `anatomy_conditioning=false`, `csme=false`, `mask_train_mode=straight_through`, seeds 42–44. True pair within an item, including the RSS twins. Do not use the existing `kspace_weight` path: that is an SSIM between the FFT of the RSS image and a coil-averaged measured k-space magnitude.

## In flight

- [ ] **LOUPE loss and partial Fourier (proposal B)** — branch `exp/loss_domain__loupe_loss_partial_fourier`
  - Renamed from `exp/loupe_loss_partial_fourier` on 2026-11-20, with its report, launch and CoreHPC folders.
  - Question: does LOUPE's learned map use partial Fourier differently under an RSS or a complex-image L1? RSS vs
    complex L1 × free vs point-symmetric map, seeds 42 and 43, FLAIR tier F (fscr1's LOUPE recipe).
  - Proposal: `reports/loss_domain__loupe_loss_partial_fourier/proposal.md`, on the branch
  - Status: the 2026-11-19 CoreHPC validation failed on a test bug on `main` (fixed in 86b9a22). Resubmitted
    2026-11-20 at run state 2 (24db7c7, which also adds the relative probe-bar addendum): validation 3161843, then the
    8 arms 3161844–3161851.

## Next

- [ ] **1. Image space: complex L1, mask on vs off, against RSS L1** — branch `exp/complex_image_mask`
  - Question: Does the image-space support mask change RSS PSNR, the LOUPE mask, or the maps' common phase for a complex-image L1? (open question 1)
  - Control: unmasked complex L1, `recon_complex` against the cached image `x`, divided by the per-example L1 norm of `x`. Same seeds also train the production RSS L1 with no `M`. That RSS arm is the loss-bar twin for this item and for items 2 and 3.
  - Treatment: the single change from the unmasked complex arm is `M`. L1 of `M · recon_complex` against `M · x`, divided by the per-example L1 norm of `M · x`.
  - Expected outcome: RSS PSNR within about ±0.2 dB of both twins. Mask-on drops the background term. The reconstruction phase should move toward the cached image on both complex arms.
  - Success criterion: mask bar on masked minus unmasked; loss bar on each complex arm minus its RSS twin. SSIM_Masked ≥ 0.88 on every run.
  - Failure means: the mask bar missed means the background term is not load-bearing; record one complex-image result. The loss bar missed means NO-GO for complex-image L1 as the default. Still record sampling-mask correlation against the three RSS-L1 seeds, and the common-phase circular std.
  - Depends on: the cache, and the overlap of `M` with the metric mask
  - Compute: 9 runs × 100 epochs, from scratch. The anatomy-pair VarNet was about 171 s/epoch (about 5 h). About 45 GPU-hours; 4 GPUs, about 15 h wall. Checkpoints in `~/data/seqmri_checkpoints`.

- [ ] **2. K-space: L1 through the learned maps, mask on vs off** — branch `exp/kspace_sense_mask`
  - Question: Does the same image-space mask change those readouts once the complex image is expanded with the learned sensitivity maps? (open question 2)
  - Control: mask-off k-space L1. `k_c = FFT(S_c · recon_complex)` against `FFT(S_c · x)`, ortho FFT, `S` from `SensetivityModel_mc`, detached on the target only. Divided by the per-example L1 norm of the target k-space. RSS twins from item 1 if their code and metric match.
  - Treatment: the single change is `M` in front of the maps. `k_c = FFT(S_c · M · image)` on the prediction and on the detached-map target. Where `M` matches the zeros of `x`, the target matches the mask-off target and only the prediction changes.
  - Expected outcome: RSS PSNR within about ±0.2 dB of the RSS twin. Mask-on vs mask-off should be smaller than in item 1 if the cached image is already zero outside `M`, because the target k-space does not change. A large gap means background energy in the prediction, after `S`, is moving the loss.
  - Success criterion: same two bars as item 1. The loss bar is k-space minus the RSS twin. The mask bar is masked minus unmasked k-space.
  - Failure means: NO-GO for this k-space L1 as the default when the loss bar is missed. A missed mask bar means the image-space mask is irrelevant once the expansion uses `S`. Record the gap versus item 1.
  - Depends on: 1 (the definition of `M`, the seeds, and the RSS twins)
  - Compute: 6 runs if the item-1 RSS twins are reused, else 9. Same cost per run as item 1.

- [ ] **3. RSS k-space: L1 of the FFT of the RSS magnitude, mask on vs off** — branch `exp/kspace_rss_mask`
  - Question: Does moving the production RSS L1 into k-space, with and without `M`, change RSS PSNR, the LOUPE mask, or the maps' common phase? (open question 3)
  - Control: mask-off RSS k-space. Ortho FFT of the same RSS magnitude the baseline uses: `FFT(rss_pred)` against `FFT(rss_target)`, complex L1, divided by the per-example L1 norm of `FFT(rss_target)`. No sensitivity maps. RSS image twins from item 1 if their code and metric match.
  - Treatment: the single change is `M` on the RSS images before the FFT. `FFT(M · rss_pred)` against `FFT(M · rss_target)`. The RSS target is not zero outside `M`, so both sides change. This is not a fit to the measured coil k-space.
  - Expected outcome: RSS PSNR within about ±0.2 dB of the RSS image twin. Unmasked L1 in k-space is not the image L1: an ortho FFT preserves L2, not L1. Common-phase circular std should stay with the RSS baseline. This loss never sees the complex-image phase, so it should not pin the maps the way items 1 and 2 can.
  - Success criterion: same two bars as item 1. The loss bar is each RSS k-space arm minus its RSS image twin. The mask bar is masked minus unmasked RSS k-space.
  - Failure means: NO-GO for RSS k-space L1 as the default when the loss bar is missed. A missed mask bar means background in the RSS magnitude is not load-bearing once the error is an FFT. A phase collapse on this arm is a bug in the loss, not a finding about the sampler. Record sampling-mask correlation against the three RSS-L1 seeds.
  - Depends on: 1 (the definition of `M`, the seeds, and the RSS twins). Independent of item 2.
  - Compute: 6 runs if the item-1 RSS twins are reused, else 9. Same cost per run as item 1. About 30 GPU-hours; 4 GPUs, about 10 h wall.

Readout of items 1 and 2, not a separate run: if a loss pins the phase and leaves RSS metrics flat, does the sampler mask still change? Judge that against the correlation among the three RSS-L1 seeds. Item 3 is the check that a phase-blind loss does not pin the phase.

## Later / unscheduled

- Train with CSME. `recon.csme` estimates each cascade's maps from that cascade's current per-coil image, starting from the zero-filled coil images, and ignores the external maps. These runs leave the flag false.
- `coil_combine=rss` as a control that removes phase from the sampler input without changing the loss.
- RSS-domain per-example NMSE, and NMSE plus SSIM, from `docs/ADAPTIVE_MASK_EXPERIMENT_PLAN.md` §6. Same domain as today, so it does not answer the phase question. Wait until items 1–3 have verdicts.
- K-space loss against the measured coil data. Items 2 and 3 do not fit the acquired samples. Item 2 expands the cached image through the learned maps. Item 3 Fourier-transforms the RSS magnitude.

## Done

- [x] **Loss domain: RSS, k-space and complex L1** — verdict: parked (2026-11-09); RSS-image L1 stays the default
  - Report: `reports/loss_domain/`
  - Branch: `exp/loss_domain` @ 8464558 (parked; report on main 9123f44)
