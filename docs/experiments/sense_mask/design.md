# Sensitivity body mask — Design

Status: draft
Last verdict: none

## Question

`SensetivityModel_mc` RSS-normalises its U-Net output and, by default, multiplies the maps by a binary body mask. Does leaving that mask off, or taking it from the U-Net output instead of the 10×10 centre, change RSS reconstruction and the maps that are learned?

## What the mask is

`sampler.sense_mask_method` is passed to `SensetivityModel_mc.forward` as `mask_method`. The 10×10 centre crop (`mask_center`, ±5 pixels) stays on in every arm. It is the U-Net's input, the same crop as fastMRI's `SensitivityModel`.

| `mask_method` | Support | What multiplies the maps |
|---|---|---|
| `pre_network` (default) | RSS of the IFFT of the 10×10 centre, above 0.1 of that slice's max | Hard zero outside that support. The support has no weights and does not depend on the U-Net. |
| `post_network` | RSS of the U-Net output, before RSS normalisation, same 0.1 threshold | Hard zero outside that support. The comparison has no gradient, so the boundary does not train. |
| `null` | none | Nothing. Maps stay RSS-normalised over the whole field of view. Hydra `null` arrives as `None`. |

The U-Net itself always sees the centre crop. RSS normalisation is over the full field of view before the multiply, so outside activations still move the interior maps through the denominator. The treatment is whether those outside values are left on the maps that VarNet and the sampler use.

`sense_mask` is stored for figures. It is not the loss and it is not the score. The score is the trainer's apodized per-anatomy metric mask. With `coil_combine=sense`, the maps' values, including anything left outside the body, are the sampler's input image.

## Fixed setup

- Dataset: local v1, FLAIR + knee-PD + knee-PDFS, 320² grid, manifest loader (`fastmri_local_v1_flair_knee`). Train on 0.5 percent of the data
- Acceleration: R=8, `sampler.budget_over=acquired`.
- Reconstructor: E2E VarNet, 12 cascades, batch size 16, 50 epochs, lr 1e-3 stepped at epoch 40. Sensitivity maps from `SensetivityModel_mc` (10×10 centre, RSS-normalised, conv upsample, full resolution). `recon.csme` stays false.
- Sampler: shared `loupe_2d`, straight-through, `anatomy_conditioning=false`, `coil_combine=sense`.
- Score, every arm: trainer apodized per-anatomy metric mask; PSNR, NMSE, SSIM_Masked. Healthy SSIM about 0.88–0.93; below 0.80 is a bug.
- Single seed is fine for now
- Noise floor: 0.3–0.5 dB seed spread of joint LOUPE+VarNet training. Judge a pair by its run-minus-twin delta.

## Decisions

| # | Decision | Why | Evidence |
|---|----------|-----|----------|
| 1 | The factor under test is `mask_method` on `SensetivityModel_mc` | The 10×10 centre, the LOUPE mask, and the apodized metric mask are three other masks. This experiment moves only the body multiply | `SensetivityModel_mc.forward`; `configs/config.yaml` has no `sense_mask_method` key, so the default is the builder's `pre_network` |
| 2 | Score with the apodized metric mask on every arm | The 2026-10-04 1-cascade comparison logged PSNR with a metric that followed the sense mask, so those PSNRs cannot be compared | `reports/PARTIAL_FOURIER_REPORT.md` addendum; metric change 2026-10-20 |

## Settled results (do not relitigate)


## Open questions

- Is masking beneficial? 
- If masking is beneficial, where is the best place to mask?

## Success rule

- Better performance of one vs the other

## Constraints

- Both arms of a seed train from scratch. The CG-SENSE nomask checkpoint is a different reconstructor, dataset, and pairing.
- Crashed runs are rerun in full. Checkpoints go to `~/data/seqmri_checkpoints`.
- No compute on the HPC login node.
- The image-space factor `M` in `docs/experiments/loss_domain/` is a weight on the loss. It is independent of this body multiply. These runs stay on `loss_type=l1`.

## Links

- Roadmap: [roadmap.md](roadmap.md)
