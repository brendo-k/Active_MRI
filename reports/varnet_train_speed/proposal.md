# Proposal: varnet_train_speed

**Status:** frozen 2026-11-02 22:45, written by Claude at the user's request: "improve the training speed of the varnet
plus loupe ... explore convergence speeds too like muon and different norm layers. Use 0.2 percent of the data for now
... Run all night on all 4 gpus until 7AM". The user asked for an overnight run, so this proposal was not approved
before launch. The user can veto any arm in the morning.

- **Topic / branch:** `varnet_train_speed`, branch `exp/varnet_train_speed`, report `reports/varnet_train_speed/`.
- **Question:** how much less wall-clock does the main 1-step LOUPE + VarNet setup (R=8) need to reach a given validation
  quality? This has two parts:
  1. **Throughput:** exact changes to the step and the trainer, giving the same training math with less time per step
     and per epoch.
  2. **Convergence:** optimizer (Muon) and normalisation-layer changes that need fewer epochs, or less time per epoch,
     to reach the control's quality.

  Why it matters: roadmap effects are 0.05–0.2 dB against 0.3–0.5 dB seed spread, so every question needs several
  seeds. A full 100-epoch run takes ~12.5 h, which caps how many seeds and arms fit.
- **Data:** `datasets=fastmri_local_v1_flair_knee`, `datasets.limit_volumes=0.2`.
  - "0.2 percent" is read as the loader's fraction 0.2 (20% of the volumes of each condition). A literal 0.2% keeps one
    volume per class.
  - Train is 16 FLAIR + 8 PD + 8 PDFS volumes (241 + 266 + 265 = 772 slices, 49 batches of 16). Val is 8 + 4 + 4
    volumes (~390 slices).
  - The volume draw is fixed (seed 1234) and is the same in every run.

## Control
- 12-cascade E2E VarNet (fastMRI U-Net, chans 18, InstanceNorm, bf16 autocast, compiled).
- `loupe_2d` straight-through mask, VD p=4 start, R=8 of the acquired k-space, 320² grid, batch 16.
- Adam lr 1e-3, StepLR ×0.1 at epoch 36 of 40 (90%, as the 100-epoch runs step at 90).
- Per-condition masks off, main @ 457c788 (snapshot `../SeqMRI-worktrees/run-vts-457c788`).
- Seeds 42, 43, 44. Logged to wandb group `varnet_train_speed` (run names `vts_<arm>_s<seed>`).

## Experiment arms (one change each against the control; same seeds, data, batch order and budget)
| arm | change |
|---|---|
| `muon` | `training.optimizer=muon`: Muon (Nesterov momentum 0.95, 5 Newton–Schulz steps, update RMS-matched to Adam, lr 1e-3, same step schedule) on the 3×3 and transposed convolutions inside every U-Net (cascades + sensitivity net). The in/out convolutions, biases, DC weights and LOUPE logits keep Adam. |
| `norm_layer` | `recon.norm=layer sampler.sens_norm=layer`: GroupNorm(1, C) with affine in place of every InstanceNorm. |
| `norm_none` | `recon.norm=none sampler.sens_norm=none`: no normalisation in the U-Net bodies. NormUnet's input normalisation stays. |
| `norm_chln` | `recon.norm=channel_layer sampler.sens_norm=channel_layer`: per-pixel LayerNorm over channels (NAFNet / Restormer style). |
| follow-ups | Decided overnight from the first results, each one change against the control: Muon at another lr, Muon + the best norm, a 2nd/3rd seed of promising arms. Each is logged here as it is launched. |

The norm swaps draw no random numbers, so every conv weight starts as in the control for the same seed. The Muon arm
starts from identical weights.

Throughput changes are exact by construction and are judged by an equivalence test plus an idle-GPU benchmark, not by
training. Candidates:
- train-batch metrics only every N steps (they are ~0.2 s of a 1.45 s step);
- cheaper validation (20 s of a 95 s epoch at this data size, 79 of 455 s at full data);
- removing host syncs and per-epoch plotting.

## Noise floor
Three control seeds. Arms get 2 seeds, or 3 when the first two disagree with each other or land within the control
spread. Comparisons are per epoch, paired by seed. The data draw, batch order and initial weights are shared within a seed.

## Metrics
- Trainer validation `val/PSNR_Masked` and `val/SSIM_Masked`, inside the apodized per-anatomy metric mask, overall and per
  anatomy, every epoch.
- `TrainTime` per epoch, and ms/step from an idle-GPU benchmark (`profile_step.py`).
- Time-to-quality = the first epoch at which an arm's val PSNR_Masked reaches the control's 3-seed mean at epoch 40,
  times the arm's epoch time.

## Expected outcomes
- **Muon:** faster early convergence, +0.2 to +0.5 dB at matched epochs before the lr step. At epoch 40, within ±0.1 dB
  of the control. Possibly unstable at lr 1e-3 with RMS matching. If so, the follow-up tries lr 3e-4.
- **norm_layer:** about equal to the control (±0.1 dB). If better, InstanceNorm's per-channel contrast removal is
  costing something.
- **norm_none:** 10–20% faster per step (the norms are ~20% of U-Net time). Worse, or unstable early, without
  normalisation at lr 1e-3.
- **norm_chln:** unknown. Restoration networks favour it, but the fastMRI U-Net was tuned with InstanceNorm.
- **Throughput:** 10–15% faster train epochs from the metrics change. More time per epoch saved by cheaper validation.

## Success criteria (fixed now)
- **Exact throughput change: adopt** if
  1. outputs and gradients match the old code (bitwise, or ≤1e-5 relative where only reduction order changes), and
  2. the benchmark step or epoch time drops by ≥5%.
- **Convergence arm: faster** if both hold:
  1. its time-to-quality is ≤75% of the control's (the control reaches its own epoch-40 mean at epoch 40 × its epoch
     time);
  2. its epoch-40 val PSNR_Masked is ≥ the control mean − 0.1 dB in every seed, with SSIM_Masked not below the control
     mean − 0.002.
- **Convergence arm: better** if its epoch-40 val PSNR_Masked beats the control mean by ≥0.1 dB in every seed, with no
  SSIM loss.
- **Worse:** epoch-40 PSNR below the control mean − 0.1 dB in every seed.
- **Inconclusive:** anything else.
- These results are on 20% of the data and a 40-epoch budget. A "faster" verdict is a candidate for the full-data check,
  not a change to the default.

## Compute plan
- One run per A6000 (~37–41 GB), ~95 s/epoch, so ~65 min per 40-epoch run.
- The queue (`scripts/launch/varnet_train_speed_worker.sh`) runs from `~/data/seqmri_checkpoints/varnet_train_speed/`
  on GPUs 0–3 until 06:50. It waits while another user holds a GPU (GPU 1 had the loss_domain eval at 22:30).
- A run that stops early is rerun from scratch, up to 3 attempts.
- About 25 GPU-hours, so ~20 runs: 3 controls, 4 arms × 2 seeds, plus follow-ups.
