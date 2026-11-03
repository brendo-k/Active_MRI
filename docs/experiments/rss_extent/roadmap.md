# RSS acquired extent — Roadmap

Generated from [design.md](design.md) on 2026-11-01. Edit freely; the loop runs the first unchecked item whose dependencies are done.

## Arms

`masked` is the control. It is the production loss image: expand the coil-combined reconstruction through the sensitivity maps, zero coil k-space outside `_kspace_extent`, RSS, L1 against the loader target. `unmasked` drops that zeroing, so the loss is the RSS of the coil images `S · x` against the same target. `output` stays the masked RSS on both arms, and the score is that image under the trainer apodized per-anatomy metric mask: PSNR, NMSE, SSIM_Masked. A true pair shares the seed, the mask draw, the reconstructor and sensitivity-net init, and the batch order. The only change is the extent multiply in the loss.

| Arm | Extent on the loss RSS | Item |
|-----|------------------------|------|
| Acquired rectangle | `masked` | control |
| Full grid | `unmasked` | 1, treatment |

Shared on every training arm: local v1 FLAIR + knee-PD + knee-PDFS, 320², R=8, `budget_over=acquired`, E2E VarNet with 12 cascades, batch size 16, 50 epochs, lr 1e-3 stepped at epoch 40, `loupe_2d`, `mask_train_mode=straight_through`, `loupe_vd_init_power=0`, `anatomy_conditioning=false`, `coil_combine=sense`, `csme=false`, `restrict_recon_to_extent=false`, `loss_type=l1`, 0.5 volumes.

Verdicts are per protocol (FLAIR, PD, PDFS). FLAIR's acquired rectangle is a smaller fraction of the grid than knee's, so a pooled PSNR can hide a split.

## Next

- [ ] **1. Does the acquired-extent mask help the RSS loss?** — branch `exp/rss_extent`
  - Question: On a matched 12-cascade VarNet pair, does `masked` beat `unmasked` on trainer PSNR? (open question 1)
  - Control: loss on the masked RSS, seed 42. This is today's `_coil_rss`.
  - Treatment: `unmasked`, paired by seed. The loss is L1 of the RSS of `S · x` with no extent multiply. The logged score is still the masked RSS. Both arms train from scratch.
  - Expected outcome: a larger gap on FLAIR than on knee. Knee's rectangle already covers most of the grid, so the two RSS images nearly agree.
  - Success criterion: one arm has higher trainer PSNR than the other, per protocol.
  - Depends on: none
  - Compute: 2 runs × 50 epochs, from scratch. At 0.5 volumes a VarNet epoch is a few minutes; about 3 h per run. Checkpoints in `~/data/seqmri_checkpoints`.
