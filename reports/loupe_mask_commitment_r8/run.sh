#!/usr/bin/env bash
# Regenerates every result file in reports/loupe_mask_commitment_r8/.
# The 8 runs trained on CoreHPC (sbatch array 3118620 from run state 0bf4cec,
# scripts/launch/hpc/loupe_mask_commitment_r8/run.sbatch). Their checkpoints, .hydra configs, logs and wandb folders were
# copied with rsync to /export/scratch/bkadota/loupe_mask_commitment_r8/runs (checkpoints/lmc_* are symlinks to them).
# The eval needs one GPU (~1.5 h for 8 runs) and reuses eval_last_model.per_slice.npz: delete it to force a rescore.
# The mask script runs on CPU and also re-scores the previous experiment's 12-cascade lr 1e-3 controls as a check.
set -euo pipefail
cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=1 python scripts/eval/varnet_loupe_lr_schedule_r8_eval.py --runs-dir /export/scratch/bkadota/loupe_mask_commitment_r8/runs --control control --checkpoint last_model.pt --data-root '/export/scratch/bkadota/fastmri_raw/{anatomy}' --out reports/loupe_mask_commitment_r8/eval_last_model.json
python scripts/plots/loupe_mask_commitment_r8_masks.py --runs /export/scratch/bkadota/loupe_mask_commitment_r8/runs --reference /export/scratch/bkadota/varnet_loupe_lr_schedule_r8/runs_full --report reports/loupe_mask_commitment_r8
CUDA_VISIBLE_DEVICES=1 python scripts/plots/loupe_mask_commitment_r8_recons.py --runs /export/scratch/bkadota/loupe_mask_commitment_r8/runs --report reports/loupe_mask_commitment_r8 --data-root '/export/scratch/bkadota/fastmri_raw/{anatomy}'
python scripts/eval/loupe_mask_commitment_r8_summary.py --runs /export/scratch/bkadota/loupe_mask_commitment_r8/runs --report reports/loupe_mask_commitment_r8
