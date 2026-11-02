# Loss domain — Design

Status: draft
Last verdict: none

## Question

Training minimizes L1 on the RSS magnitude. VarNet already emits a coil-combined complex image, and a cached ESPIRiT image supplies a complex target. Does that loss, and the same error after expansion through the learned sensitivity maps into k-space, change the LOUPE mask, the maps, and the RSS metrics? The image-space support mask is tested on and off in both domains.

## Fixed setup

- Dataset: local v1, FLAIR + knee-PD + knee-PDFS, 320² grid, manifest loader.
- Complex target: the cache at `/export/scratch/bkadota/fastmri_complex_espirit`, one ESPIRiT coil-combined image per slice. Presets in `activemri/utils/bart_ecalib.py`. Maps were not stored.
- Acceleration: R=8 with `sampler.budget_over=acquired` set explicitly. The shared launcher comment still describes grid-R; this feature does not use that.
- Sampler: shared `loupe_2d`, straight-through, VD start `loupe_vd_init_power=4`, `anatomy_conditioning=false`. `sampler.coil_combine` stays `sense`, so the sampler still sees the maps' phase.
- Reconstructor: E2E VarNet, 12 cascades, batch size 16, 100 epochs, lr 1e-3 stepped at epoch 90. Sensitivity maps from `SensetivityModel_mc` (10×10 ACS, RSS-normalised). `recon.csme` stays false, so these runs keep that one external map set.
- Image-space mask `M`: 1 where the cached image has `|x| > 0`, else 0. Recomputed from the cache, not stored. The mask-off arms leave this factor at 1.
- Score, every arm: trainer apodized per-anatomy metric mask; PSNR, NMSE, SSIM_Masked. The training loss may change; the score does not. Healthy SSIM about 0.88–0.93; below 0.80 is a bug.
- Pairs: same seed, same mask draw, same reconstructor and sensitivity-net init, same batch order. Seeds 42, 43, 44.
- Noise floor: 0.3–0.5 dB seed spread of joint LOUPE+VarNet. Sampling-mask change is judged against the correlation among the three RSS-L1 seeds, not against 1.0.

## Decisions

| # | Decision | Why | Evidence |
|---|----------|-----|----------|
| 1 | Keep scoring RSS PSNR / NMSE / SSIM_Masked on every arm | The losses change the objective; the comparison has to stay on one metric | `AGENTS.md` metric gotcha; `docs/ADAPTIVE_MASK_EXPERIMENT_PLAN.md` §6 |
| 2 | The complex target is the cached ESPIRiT image | There is no complex image in the fastMRI files. The cache was built on the loader-normalized k-space and the maps were discarded | `scripts/data/build_complex_image_cache.py`; 480 volumes, 10,270 slices, 5.1 GB |
| 3 | Test the image-space mask on and off | `M` drops the background, where the ESPIRiT crop is already zero. Mask-on and mask-off are the comparison, so the background term is measured instead of being built into every arm | cache images are zero outside the eigenvalue crop |
| 4 | K-space loss masks the complex image, then expands with the learned maps | `k_c = FFT(S_c · M · image)` with the ortho FFT, `S` from `SensetivityModel_mc`. Mask-off uses `M = 1`. The target uses `S` detached, so the loss cannot shrink the error by editing the reference. The prediction uses the live maps | `VarNetBlock.sens_expand` is `FFT(S · image)` |
| 5 | Divide each arm's L1 by the per-example L1 norm of its target | Mask-on removes pixels, and k-space L1 lives on another scale. Either one would change the gradient size as well as the loss | |
| 6 | The RSS arm stays the production L1, with no image-space mask | It is the control. `M` is a factor on the complex and k-space arms | current `loss_type=l1` |
| 7 | Leave `coil_combine=sense` | The sampler should still see the maps' phase. `rss` would hide that phase | `sampler.coil_combine` in `configs/config.yaml` |

## Settled results (do not relitigate)

None yet.

## Open questions

- On the complex-image loss, does `M` move RSS PSNR, the LOUPE mask, or the maps' common phase relative to the unmasked complex arm and to RSS L1? Blocks roadmap item 1.
- On the k-space loss `FFT(S · M · image)`, does the same `M` move those readouts? Blocks roadmap item 2. Where `M` matches the zeros of the cached image, the target `FFT(S · M · x)` equals `FFT(S · x)`. The mask then changes only the prediction, by dropping background before the coil expansion.
- On an RSS k-space loss, the ortho FFT of the production RSS magnitude, does `M` move RSS PSNR, the LOUPE mask, or the maps' common phase relative to the unmasked RSS k-space arm and to RSS L1? Blocks roadmap item 3. Same images as the baseline; only the domain of the L1 changes. No expansion through `S`, and no fit to the measured coil k-space. This loss does not see the complex-image phase, so a phase change here is a surprise.
- How much of `M` overlaps the apodized metric mask? The score uses the metric mask. A large mismatch means the score is looking where a mask-on loss was shut off. Report it with item 1. It does not move the success bar.
- If a loss pins the phase and leaves RSS metrics flat, does the sampler mask still change? A readout of items 1 and 2, judged against the correlation among the three RSS-L1 seeds.

## Success rule

Pre-register per experiment, before the run. Two bars, both on RSS PSNR under the apodized metric mask, three seeds, SSIM_Masked ≥ 0.88 on every run.

- Mask bar: masked minus unmasked mean ≥ 0.10 dB in absolute value, same sign in all 3 seeds.
- Loss bar: that arm minus its RSS-L1 twin mean ≥ +0.10 dB, same sign in all 3 seeds.

Sampling-mask correlation and common-phase circular standard deviation are reported for every seed and do not gate either bar. A phase or sampling-mask change inside ±0.10 dB is a finding and not a reason to change the default loss.

## Constraints

- Both arms of a seed train from scratch. Do not score an older L1 checkpoint against a new loss: the sampler-input conj bug (2026-10-27) and the metric-mask date make old runs a different experiment.
- Crashed runs are rerun in full. Checkpoints go to `~/data/seqmri_checkpoints`.
- No compute on the HPC login node.
- The REINFORCE reward path ignores `loss_type`. These runs stay on `mask_train_mode=straight_through`.
- The complex-image cache lives on scratch, not in the repo and not under `/home`.

## Links

- Roadmap: [roadmap.md](roadmap.md)
- Related docs: `docs/ADAPTIVE_MASK_EXPERIMENT_PLAN.md` §6, `docs/features/anatomy_routing/roadmap.md` (complex-image or k-space loss listed as later), `AGENTS.md`
- Reports: none yet
