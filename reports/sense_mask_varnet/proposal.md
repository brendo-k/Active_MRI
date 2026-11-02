# Proposal: sense_mask_varnet (exploratory, 1 seed)

Branch `exp/sense_mask_varnet`, frozen 2026-11-01. Roadmap: `docs/experiments/sense_mask/` (items 1 and 3).

## Question
On a 12-cascade E2E VarNet with a jointly trained LOUPE mask, does the hard body multiply on the `SensetivityModel_mc` maps help, and does it matter where its support comes from?

## Arms (the only difference is `sampler.sense_mask_method`)
| Arm | `sense_mask_method` | Multiplies the maps by |
|---|---|---|
| control | `pre_network` | hard zero outside RSS of the 10x10 centre IFFT, above 0.1 of the slice max |
| no mask | `null` | nothing; maps stay RSS-normalised over the whole FOV |
| U-Net mask | `post_network` | hard zero outside RSS of the U-Net output (before RSS norm), same 0.1 threshold |

## Shared setup
Local v1 FLAIR + knee-PD + knee-PDFS, 320x320, manifest loader (`fastmri_local_v1_flair_knee`), `limit_volumes=0.5`.
R=8, `budget_over=acquired`. E2E VarNet, 12 cascades, `csme=false`, batch 16, 50 epochs, lr 1e-3 stepped at epoch 40,
`loss_type=l1`. Sampler `loupe_2d`, `mask_train_mode=straight_through`, `loupe_vd_init_power=0`,
`anatomy_conditioning=false`, `coil_combine=sense`. Seed 42 in every arm: same init, mask draw and batch order.
All arms train from scratch. Score: trainer apodized per-anatomy metric mask (PSNR, NMSE, SSIM_Masked), per protocol.

## Noise floor
Joint LOUPE+VarNet seed spread is 0.3-0.5 dB. One seed per arm, so this is **exploratory**: no difference can be called
significant, and the result only decides whether a multi-seed follow-up is worth running.

## Expected outcome
Small differences, within +-0.2 dB PSNR, with NMSE and SSIM_Masked moving the same way.
- `null` better: the body multiply removes signal the reconstruction uses.
- `pre_network` better: the multiply suppresses out-of-body noise in the maps.
- `post_network` vs `pre_network`: the U-Net support has no gradient at its boundary, so it can only differ by where the cut falls.
- All within noise: the multiply does not matter at 12 cascades.

## Criteria (frozen)
- **Signal worth a multi-seed follow-up:** an arm differs from control by >= 0.5 dB PSNR, in the same direction on at least 2 of the 3 protocols (FLAIR, PD, PDFS).
- **Inconclusive:** every |delta PSNR| < 0.5 dB, or a difference present on only 1 protocol.
- **Invalid run:** SSIM_Masked < 0.80 or a crash; debug, then rerun in full.

## Compute
3 runs x 50 epochs on one H100 NVL each, from scratch. About 171 s/epoch for a comparable VarNet, so about 2.4 h each.
Checkpoints under `/home/remote/bkadota/scratch/logs/active_mri/`, wandb offline.
