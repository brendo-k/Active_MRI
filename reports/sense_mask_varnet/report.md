# Does the body multiply on the sensitivity maps help a 12-cascade VarNet, and where should its support come from? (2026-11-01)

| | |
|---|---|
| Code | results produced at `975316e` (sense_mask_varnet: script for the final validation metrics of the offline wandb runs); branch `exp/sense_mask_varnet` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 3 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 4 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |

**Status: exploratory, 1 seed. Awaiting verdict.**
**Branch**: `exp/sense_mask_varnet`. The three runs trained at commit `955bfb1` (recorded in each run's `run_info.json`); the results script is `975316e`.

## Proposal vs outcome

Criteria frozen in `proposal.md` (commit `002505d`), not edited since.

| Frozen criterion | Result | Status |
|---|---|---|
| An arm differs from control by >= 0.5 dB PSNR, same direction on >= 2 of 3 protocols (FLAIR, PD, PDFS) | `null`: +1.26 dB PSNR_Masked pooled, +2.22 dB brain, +1.98 dB knee. `post_network`: +0.67 dB pooled, +1.56 dB brain, +0.93 dB knee | **Cannot be evaluated as written.** The trainer logs brain and knee, not knee-PD and knee-PDFS. Brain (FLAIR) passes for both arms; the knee value pools PD and PDFS. |
| Invalid run: SSIM_Masked < 0.80 or a crash | all three completed 50/50 epochs; SSIM_Masked 0.815 / 0.828 / 0.840 | not triggered, but see Caveats: all are below the 0.88-0.93 the design doc calls healthy |

**Recommendation** (the verdict is yours): the signal is large against the 0.3-0.5 dB seed spread, and every pooled metric orders the arms the same way, so a multi-seed follow-up looks worth running. I would first score the PD/PDFS split from the saved checkpoints so the frozen criterion can be decided as written, and I would not call the mechanism settled (see Caveats on the training loss). Merge or park after that; I have not done either.

## Question

`SensetivityModel_mc` RSS-normalises its U-Net output and, by default, multiplies the maps by a hard body mask. On a
12-cascade E2E VarNet trained jointly with a LOUPE mask, does that multiply help, and does it matter whether its support
comes from the 10x10 centre (`pre_network`) or from the U-Net output (`post_network`)? Convention: PSNR in dB; NMSE and
SSIM unitless; all values are the last (50th) epoch over the validation set.

## Runs compared

| Arm | `sampler.sense_mask_method` | Multiplies the maps by | Job | Checkpoint dir (untracked) |
|---|---|---|---|---|
| control | `pre_network` | hard zero outside RSS of the 10x10 centre IFFT, > 0.1 of the slice max | 2863425 | `~/scratch/logs/active_mri/sense_mask_varnet_pre_network_seed42` |
| no mask | `null` | nothing | 2863426 | `.../sense_mask_varnet_null_seed42` |
| U-Net mask | `post_network` | hard zero outside RSS of the U-Net output, same threshold | 2863427 | `.../sense_mask_varnet_post_network_seed42` |

Shared: local v1 FLAIR + knee-PD + knee-PDFS, 320x320, `limit_volumes=0.5`, R=8, `budget_over=acquired`, E2E VarNet with
12 cascades (`csme=false`), batch 16, 50 epochs, lr 1e-3 stepped at epoch 40, `loss_type=l1`, `loupe_2d` straight-through,
`loupe_vd_init_power=0`, `coil_combine=sense`, seed 42. Each run took about 1.6 h at about 107 s/epoch (80 s train, 27 s
validation) on one GPU (`--gres=gpu:1`; the GPU model was not recorded). Full command: `launch.sh`.

Validation: brain n=304 slices, knee n=657 slices (knee = PD + PDFS pooled).

## Methods

1. Train the three arms from scratch with `launch.sh` (the only override that differs is `+sampler.sense_mask_method`; the key is appended because `configs/config.yaml` does not define it).
2. Read each run's final validation epoch from its offline wandb file with `scripts/eval/wandb_final_val_metrics.py` and subtract the control. No model is re-evaluated.
3. `PSNR_Masked`, `SSIM_Masked`, `NMSE` are pooled over the validation set under the apodized per-anatomy metric mask. The brain and knee columns are the trainer's per-anatomy values, which appear to be the unmasked PSNR/NMSE (they match `val/PSNR`, not `val/PSNR_Masked`, in size); this was not checked against the trainer code.

## Results

Final epoch (50/50), seed 42:

| arm | PSNR_Masked | SSIM_Masked | NMSE | brain PSNR | knee PSNR | brain NMSE | knee NMSE |
|---|---|---|---|---|---|---|---|
| pre_network (control) | 31.26 | 0.8154 | 0.0127 | 35.79 | 28.99 | 0.0105 | 0.0136 |
| null | 32.52 | 0.8395 | 0.0068 | 38.01 | 30.97 | 0.0057 | 0.0073 |
| post_network | 31.92 | 0.8281 | 0.0088 | 37.35 | 29.92 | 0.0070 | 0.0097 |

Arm minus control (negative NMSE is better):

| arm | PSNR_Masked | SSIM_Masked | NMSE | brain PSNR | knee PSNR | brain NMSE | knee NMSE |
|---|---|---|---|---|---|---|---|
| null | +1.26 | +0.0242 | -0.0059 | +2.22 | +1.98 | -0.0048 | -0.0064 |
| post_network | +0.67 | +0.0127 | -0.0038 | +1.56 | +0.93 | -0.0035 | -0.0040 |

Every metric orders the arms `null` > `post_network` > `pre_network`, on brain and on knee. The `null` gain over control
(1.26 dB pooled) is 2.5-4x the 0.3-0.5 dB seed spread quoted in the proposal; the `post_network` gain (0.67 dB) is about
1.3-2x it. That spread comes from earlier joint-training runs, not from this configuration, and no second seed was run.

## Conclusions

- Measured: with one seed, removing the body multiply (`null`) scored the highest on every logged metric, by 1.26 dB pooled PSNR_Masked over the production `pre_network` multiply, and the U-Net-derived support (`post_network`) sat in between.
- Measured: the ordering is the same for brain and for knee.
- Not established: the frozen per-protocol criterion (FLAIR / PD / PDFS), because the PD/PDFS split was not logged.
- Hypothesis, untested: the multiply removes signal the reconstruction would otherwise use, or constrains the maps in a way that hurts joint training. Nothing here distinguishes these from the loss-region effect in Caveats.
- Hypothesis, untested: `post_network` falls between the two because its support is larger or noisier than the centre-derived one.

## Caveats

- **One seed per arm.** The proposal called this exploratory: no difference can be called significant, and seed spread for this setup was not measured.
- **Training loss differs 2.7x between arms** (final epoch TrainLoss 0.063 `pre_network`, 0.051 `post_network`, 0.023 `null`; ValLoss 0.085 / 0.069 / 0.031). The design doc says the sense mask is not part of the loss, yet the loss moves with it. If the L1 loss is computed against a target or prediction that also passes through the body mask, the arms optimise different objectives and the gap is not purely a property of the maps. Not investigated.
- **SSIM_Masked is low for all arms** (0.815 / 0.828 / 0.840). The design doc quotes a healthy 0.88-0.93 and a bug threshold of 0.80; these sit between the two. Plain `val/SSIM` is 0.925-0.934, which is healthy, and the differences between arms are the same sign. The 0.80 rule was not triggered, but the gap to 0.88 was not explained.
- **`limit_volumes=0.5`** is a fraction (half of each condition's 80 training volumes), my reading of the roadmap's "0.5 volumes". The design doc's "0.5 percent" was not taken literally.
- Per-anatomy brain/knee values may be unmasked (Methods item 3).
- Only the last epoch is reported. `best_model.pt` was not scored separately.
- The first two submissions failed on configuration errors (Hydra quoting of the data root, then the unappended `sense_mask_method` key) before any training; they produced no results and are not part of this comparison.
- Checkpoints are untracked; the manifest records overrides, config hash and wandb ids.

## Figures

None.

## Next steps

1. Score the three saved checkpoints per protocol (knee-PD and knee-PDFS separately) under the apodized metric mask, so the frozen criterion can be decided as written.
2. If `null` still leads, add seeds 43 and 44 for `pre_network` and `null` (about 1.6 h each) to get a seed spread for this configuration.
3. Find out why the training loss depends on `sense_mask_method` (Caveats) before attributing the gain to the maps.
4. Explain SSIM_Masked 0.82-0.84 against the 0.88-0.93 expected range.

## Reproduce

`bash reports/sense_mask_varnet/run.sh` regenerates `metrics.json` and `table.md` from the three offline wandb runs (CPU, seconds). It needs the run directories under `~/scratch/logs/active_mri/`. Retraining uses `launch.sh` via sbatch (one GPU, about 1.6 h per arm). Check the report with `python scripts/report_manifest.py check reports/sense_mask_varnet`.
