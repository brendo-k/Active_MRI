#!/usr/bin/env bash
# Regenerates every result file in reports/loupe_mask_commitment_r8/.
# Round 1: the 8 runs trained on CoreHPC (sbatch array 3118620 from run state 0bf4cec,
# scripts/launch/hpc/loupe_mask_commitment_r8/run.sbatch) and were copied with rsync to
# /export/scratch/bkadota/loupe_mask_commitment_r8/runs (checkpoints/lmc_* are symlinks to them).
# Follow-up: runs_e100 (run state 1bf63fd, scripts/launch/local/loupe_mask_commitment_r8_followup_e100.sh; control_s42/s43 link
# to round 1's) and runs_e150 (run state bc30363, scripts/launch/local/loupe_mask_commitment_r8_followup_e150.sh).
# Each eval needs one GPU (~45 min per 6 runs) and reuses its .per_slice.npz: delete it to force a rescore.
# The mask scripts run on CPU; the first also re-scores the previous experiment's 12-cascade lr 1e-3 controls as a check.
# Round 1's clamp and anneal checkpoints need the parked code on the branch exp/loupe_training__loupe_mask_commitment_r8.
set -euo pipefail
cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=1 python scripts/eval/varnet_loupe_lr_schedule_r8_eval.py --runs-dir /export/scratch/bkadota/loupe_mask_commitment_r8/runs --control control --checkpoint last_model.pt --data-root '/export/scratch/bkadota/fastmri_raw/{anatomy}' --out reports/loupe_mask_commitment_r8/eval_last_model.json
python scripts/plots/loupe_mask_commitment_r8_masks.py --runs /export/scratch/bkadota/loupe_mask_commitment_r8/runs --reference /export/scratch/bkadota/varnet_loupe_lr_schedule_r8/runs_full --report reports/loupe_mask_commitment_r8
CUDA_VISIBLE_DEVICES=1 python scripts/plots/loupe_mask_commitment_r8_recons.py --runs /export/scratch/bkadota/loupe_mask_commitment_r8/runs --report reports/loupe_mask_commitment_r8 --data-root '/export/scratch/bkadota/fastmri_raw/{anatomy}'
python scripts/eval/loupe_mask_commitment_r8_summary.py --runs /export/scratch/bkadota/loupe_mask_commitment_r8/runs --report reports/loupe_mask_commitment_r8
CUDA_VISIBLE_DEVICES=0 python scripts/eval/varnet_loupe_lr_schedule_r8_eval.py --runs-dir /export/scratch/bkadota/loupe_mask_commitment_r8/runs_e100 --control control --checkpoint last_model.pt --data-root '/export/scratch/bkadota/fastmri_raw/{anatomy}' --out reports/loupe_mask_commitment_r8/eval_followup_e100.json
CUDA_VISIBLE_DEVICES=0 python scripts/eval/varnet_loupe_lr_schedule_r8_eval.py --runs-dir /export/scratch/bkadota/loupe_mask_commitment_r8/runs_e150 --control control --checkpoint last_model.pt --data-root '/export/scratch/bkadota/fastmri_raw/{anatomy}' --out reports/loupe_mask_commitment_r8/eval_followup_e150.json
python scripts/plots/loupe_mask_commitment_r8_masks.py --runs /export/scratch/bkadota/loupe_mask_commitment_r8/runs_e100 --arms control mixdecay --seeds 42 43 44 --name masks_followup_e100 --report reports/loupe_mask_commitment_r8
python scripts/plots/loupe_mask_commitment_r8_masks.py --runs /export/scratch/bkadota/loupe_mask_commitment_r8/runs_e150 --arms control mix --seeds 42 43 44 --name masks_followup_e150 --report reports/loupe_mask_commitment_r8
CUDA_VISIBLE_DEVICES=0 python scripts/plots/loupe_mask_commitment_r8_recons.py --runs /export/scratch/bkadota/loupe_mask_commitment_r8/runs_e100 --report reports/loupe_mask_commitment_r8 --data-root '/export/scratch/bkadota/fastmri_raw/{anatomy}' --arms control mixdecay --eval eval_followup_e100 --seed 43 --name recons_followup_e100_s43
CUDA_VISIBLE_DEVICES=0 python scripts/plots/loupe_mask_commitment_r8_recons.py --runs /export/scratch/bkadota/loupe_mask_commitment_r8/runs_e150 --report reports/loupe_mask_commitment_r8 --data-root '/export/scratch/bkadota/fastmri_raw/{anatomy}' --arms control mix --eval eval_followup_e150 --seed 43 --name recons_followup_e150_s43
python scripts/eval/loupe_mask_commitment_r8_followup_summary.py --report reports/loupe_mask_commitment_r8 --runs-e100 /export/scratch/bkadota/loupe_mask_commitment_r8/runs_e100 --runs-e150 /export/scratch/bkadota/loupe_mask_commitment_r8/runs_e150
