# Sensitivity body mask — Roadmap

Generated from [design.md](design.md) on 2026-11-01. Edit freely; the loop runs the first unchecked item whose dependencies are done.

## Arms

`pre_network` is the control. It is the production multiply: RSS of the 10×10 centre, threshold 0.1, hard zero outside that support. The 10×10 centre crop stays on in every arm. The score on every arm is the trainer apodized per-anatomy metric mask: PSNR, NMSE, SSIM_Masked. A true pair shares the seed, the mask draw, the reconstructor and sensitivity-net init, and the batch order. The only change is `sense_mask_method`.

| Arm | `sense_mask_method` | What multiplies the maps | Item |
|-----|---------------------|--------------------------|------|
| Centre support | `pre_network` | hard zero outside the centre RSS | control |
| No body multiply | `null` | RSS-normalised maps over the whole field of view | 1, treatment |
| U-Net support | `post_network` | hard zero outside the U-Net RSS, same 0.1 threshold | 3, treatment |

Shared on every training arm: local v1 FLAIR + knee-PD + knee-PDFS, 320², R=8, `budget_over=acquired`, E2E VarNet with 12 cascades, batch size 16, 50 epochs, lr 1e-3 stepped at epoch 40, `loupe_2d`, `mask_train_mode=straight_through`, `loupe_vd_init_power=0`, `anatomy_conditioning=false`, `coil_combine=sense`, `csme=false`, `loss_type=l1`, 0.5 volumes

Map readout on every training seed: `|S|` inside the `pre_network` support of the data, so every arm is compared on that same region. Energy outside that support is a separate number. It is zero for `pre_network`.

Verdicts are per protocol (FLAIR, PD, PDFS). The settled CG-SENSE pair already splits brain from knee, and a pooled PSNR would hide that split.

## Next

- [ ] **1. Is the body multiply beneficial?** — branch `exp/sense_mask_varnet`
  - Question: On a matched 12-cascade VarNet pair, does `pre_network` beat `null` on trainer PSNR? (open question 1). This is the pair the settled CG-SENSE and 1-cascade results name as their reopen condition. Those numbers stay CG-SENSE and 1-cascade results.
  - Control: `sense_mask_method=pre_network`, seeds 42
  - Treatment: `sense_mask_method=null`, paired by seed. Both arms train from scratch.
  - Expected outcome: small movement for each. Different masks
  - Success criterion: Given a masking method, has better performance..
  - Depends on: none
  - Compute: 3 runs × 50 epochs, from scratch. The anatomy-pair VarNet was about 171 s/epoch. Checkpoints in `~/data/seqmri_checkpoints`.


