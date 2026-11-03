# RSS acquired extent — Design

Status: draft
Last verdict: none

## Question

The training loss is L1 on an image-space RSS. That RSS is formed by expanding the coil-combined image through the sensitivity maps, zeroing each coil's k-space outside the acquired rectangle, and taking the RSS of the inverse FFT. Does leaving that zeroing off change the reconstruction the trainer scores?

## What the mask is

`_coil_rss` builds the image in `pred_dict['output']`. The rectangle is `_kspace_extent`: every row and every column that holds signal in any coil. fastMRI FLAIR leaves part of the 320² grid empty (partial Fourier, and a short readout after the crop). Knee fills more of it. The loader target is the RSS of k-space that is already zero outside that rectangle, so applying the rectangle to the target again changes nothing.

| Arm | Extent on the prediction | RSS the loss sees |
|---|---|---|
| `masked` (default) | `FFT(S · x) * extent`, then IFFT | Matches the loader target when `x` is the image that produced the file. |
| `unmasked` | no multiply | RSS of the coil images `S · x`. A true image does not match the loader target. |

The between-step k-space (`FFT(image · S) * extent`, written for the next sampler step) is the same rectangle. These runs are one LOUPE step, so that write is never read. It stays masked. `restrict_recon_to_extent` is the regularizer's write region inside VarNet and stays false.

`output` stays the masked RSS on both arms. The trainer scores `output` under the apodized metric mask, so the unmasked arm trains on a second RSS and is scored on the masked one. Otherwise the two PSNRs would be different images.

## Fixed setup

- Dataset: local v1, FLAIR + knee-PD + knee-PDFS, 320² grid, manifest loader (`fastmri_local_v1_flair_knee`). Train on 0.5 volumes
- Acceleration: R=8, `sampler.budget_over=acquired`.
- Reconstructor: E2E VarNet, 12 cascades, batch size 16, 50 epochs, lr 1e-3 stepped at epoch 40. Sensitivity maps from `SensetivityModel_mc` (10×10 centre, RSS-normalised, conv upsample, full resolution). `recon.csme` stays false. `restrict_recon_to_extent` stays false.
- Sampler: shared `loupe_2d`, straight-through, `loupe_vd_init_power=0`, `anatomy_conditioning=false`, `coil_combine=sense`.
- Loss: `loss_type=l1` against the loader RSS target on both arms. The only change is whether the prediction RSS is masked to the acquired rectangle.
- Score, every arm: masked RSS under the trainer apodized per-anatomy metric mask; PSNR, NMSE, SSIM_Masked. Healthy SSIM about 0.88–0.93; below 0.80 is a bug.
- Single seed is fine for now
- Noise floor: 0.3–0.5 dB seed spread of joint LOUPE+VarNet training. Judge a pair by its run-minus-twin delta.

## Decisions

| # | Decision | Why | Evidence |
|---|----------|-----|----------|
| 1 | The factor under test is the extent multiply inside `_coil_rss` | The sampler budget, the VarNet regularizer support, and the sensitivity body mask are three other masks. This experiment moves only the rectangle in the loss RSS | `SequentialSampling._coil_rss`; `tests/test_kspace_extent.py` |
| 2 | Score the masked RSS on both arms | The trainer metrics follow `output`. An unmasked RSS of a true image does not equal the loader target, so scoring it would mix a definition change into the PSNR | `tests/test_kspace_extent.py` `test_unprojected_output_does_not`; trainer applies the metric mask to `pred_dict['output']` |
| 3 | The target stays the loader RSS on both arms | The file k-space is already zero outside the acquired rectangle. There is no second target to unmask | `FastMRIDataset.__getitem__` |

## Settled results (do not relitigate)


## Open questions

- Does masking the prediction's coil k-space to the acquired rectangle before the RSS loss help?
- Is the gap larger on FLAIR, where the rectangle covers less of the 320² grid, than on knee?

## Success rule

- Better trainer PSNR of one arm than the other, per protocol (FLAIR, PD, PDFS)

## Constraints

- Both arms of a seed train from scratch.
- Crashed runs are rerun in full. Checkpoints go to `~/data/seqmri_checkpoints`.
- No compute on the HPC login node.
- The sensitivity body multiply in `docs/experiments/sense_mask/` and the image-space factor `M` in `docs/experiments/loss_domain/` are different masks. These runs keep `sense_mask_method` at its default and stay on `loss_type=l1`.

## Links

- Roadmap: [roadmap.md](roadmap.md)
