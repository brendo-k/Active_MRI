#!/usr/bin/env bash
# Regenerates every result file in reports/varnet_loupe_lr_schedule_r8/.
# The 49 runs trained on CoreHPC (first sweep: sbatch array 3042468 from run state 666f872,
# scripts/launch/hpc/varnet_loupe_lr_schedule_r8/run.sbatch; follow-up: array 3053049 from 15680ca, followup.sbatch;
# follow-up 2: array 3080452 from 936f3f6, CASCADES=6 followup.sbatch).
# Their checkpoints, .hydra configs, logs and wandb folders were copied with rsync to
#   /export/scratch/bkadota/varnet_loupe_lr_schedule_r8/{runs,runs_full,runs_full_c6}
# (checkpoints/vlls_*, vlls_full_* and vlls_full_c6_* are symlinks to them). The evals read the local copy of local v1
# through --data-root. The scoring itself is varnet_train_speed_eval.score (imported). Each eval needs one GPU, about 1 h for 33 runs; it reuses eval_*.per_slice.npz, so delete
# those to force a rescore.
set -euo pipefail
cd "$(dirname "$0")/../.."

# First sweep (20% data, 400 epochs): last_model.pt (the frozen score) and best_model.pt, on the full local v1 val
CUDA_VISIBLE_DEVICES=1 python scripts/eval/varnet_loupe_lr_schedule_r8_eval.py --runs-dir /export/scratch/bkadota/varnet_loupe_lr_schedule_r8/runs --checkpoint last_model.pt --data-root '/export/scratch/bkadota/fastmri_raw/{anatomy}' --out reports/varnet_loupe_lr_schedule_r8/eval_runs_last_model.json
CUDA_VISIBLE_DEVICES=1 python scripts/eval/varnet_loupe_lr_schedule_r8_eval.py --runs-dir /export/scratch/bkadota/varnet_loupe_lr_schedule_r8/runs --checkpoint best_model.pt --data-root '/export/scratch/bkadota/fastmri_raw/{anatomy}' --out reports/varnet_loupe_lr_schedule_r8/eval_runs_best_model.json
# Follow-up (full data, 100 epochs): last_model.pt
CUDA_VISIBLE_DEVICES=1 python scripts/eval/varnet_loupe_lr_schedule_r8_eval.py --runs-dir /export/scratch/bkadota/varnet_loupe_lr_schedule_r8/runs_full --checkpoint last_model.pt --data-root '/export/scratch/bkadota/fastmri_raw/{anatomy}' --out reports/varnet_loupe_lr_schedule_r8/eval_runs_full_last_model.json
# Follow-up 2 (6 cascades, full data, 100 epochs): last_model.pt
CUDA_VISIBLE_DEVICES=1 python scripts/eval/varnet_loupe_lr_schedule_r8_eval.py --runs-dir /export/scratch/bkadota/varnet_loupe_lr_schedule_r8/runs_full_c6 --checkpoint last_model.pt --data-root '/export/scratch/bkadota/fastmri_raw/{anatomy}' --out reports/varnet_loupe_lr_schedule_r8/eval_runs_full_c6_last_model.json
# LOUPE mask comparison (post hoc, CPU): masks.json, figures/masks.png, figures/masks_over_training.png
python scripts/plots/varnet_loupe_lr_schedule_r8_masks.py --runs /export/scratch/bkadota/varnet_loupe_lr_schedule_r8 --report reports/varnet_loupe_lr_schedule_r8
# Frozen criteria of the three proposals, twin contrasts, trainer val-curve peaks
python scripts/eval/varnet_loupe_lr_schedule_r8_summary.py --runs /export/scratch/bkadota/varnet_loupe_lr_schedule_r8 --report reports/varnet_loupe_lr_schedule_r8
