#!/usr/bin/env bash
# Regenerates every result file of reports/brain_varnet_width from the 13 new runs in
# ~/data/seqmri_checkpoints/brain_varnet_width (linked as checkpoints/bvw_*) and the 4 reused 80-volume runs of
# brain_varnet_scaling (checkpoints/bvs_c{6,12}_v80_s{42,43}).
# The eval script skips runs whose score file exists: delete reports/brain_varnet_width/eval/{val,train}_*.pt to recompute.
# Training itself: scripts/launch/queue_brain_varnet_width.sh (seed 42 "all", then seed 43 with 3:36 6:26 12:12).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH=scripts/diagnostics

SCALING=/home/bkadota/code/active_mri/SeqMRI/reports/brain_varnet_scaling/eval
for split in val train; do for ckpt in best_model last_model; do for seed in 42 43; do for c in 6 12; do
    cp -n "$SCALING/${split}_${ckpt}_c${c}_v80_s${seed}.pt" "reports/brain_varnet_width/eval/${split}_${ckpt}_c${c}_ch18_s${seed}.pt"
done; done; done; done

CUDA_VISIBLE_DEVICES=0 python scripts/eval/brain_varnet_width_eval.py --seed 42 --split val
CUDA_VISIBLE_DEVICES=0 python scripts/eval/brain_varnet_width_eval.py --seed 42 --split train
CUDA_VISIBLE_DEVICES=0 python scripts/eval/brain_varnet_width_eval.py --seed 43 --split val
CUDA_VISIBLE_DEVICES=0 python scripts/eval/brain_varnet_width_eval.py --seed 43 --split train
python scripts/eval/brain_varnet_width_summary.py

# Tier L (proposal_tier_l.md): CoreHPC array job 3079379 trained scripts/launch/hpc/brain_varnet_width_l/run.sbatch
# (sbatch --array=0-9). Copy the runs back, then score with the local data copies.
# Delete reports/brain_varnet_width/eval_{L,hpcS}/*.pt to recompute.
# rsync -a --exclude 'visualizations*' corehpc:scratch/brain_varnet_width_l/runs_{L,S} ~/data/seqmri_checkpoints/brain_varnet_width_l/
L=/home/bkadota/data/seqmri_checkpoints/brain_varnet_width_l
for split in val train; do
    CUDA_VISIBLE_DEVICES=1 python scripts/eval/brain_varnet_width_eval.py --local-data --queue $L/runs_L --seed 42 43 --split $split --out reports/brain_varnet_width/eval_L
    CUDA_VISIBLE_DEVICES=1 python scripts/eval/brain_varnet_width_eval.py --local-data --queue $L/runs_S --seed 42 43 --split $split --out reports/brain_varnet_width/eval_hpcS
done
PYTHONPATH=scripts/diagnostics:scripts/eval python scripts/eval/brain_varnet_width_tier_l_summary.py
