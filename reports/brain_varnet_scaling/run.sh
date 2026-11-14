#!/usr/bin/env bash
# Regenerates every result file of reports/brain_varnet_scaling from the 21 trained runs in
# ~/data/seqmri_checkpoints/brain_varnet_scaling (linked as checkpoints/bvs_*).
# The eval script skips runs whose score file exists: delete reports/brain_varnet_scaling/eval/{val,train}_*.pt to recompute.
# Training itself: scripts/launch/queue_brain_varnet_scaling.sh (seed 42, then SEED=43 with the frozen picks).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH=scripts/diagnostics

CUDA_VISIBLE_DEVICES=0 python scripts/eval/brain_varnet_scaling_eval.py --seed 42 --split val
CUDA_VISIBLE_DEVICES=0 python scripts/eval/brain_varnet_scaling_eval.py --seed 42 --split train
CUDA_VISIBLE_DEVICES=0 python scripts/eval/brain_varnet_scaling_eval.py --seed 43 --split val
CUDA_VISIBLE_DEVICES=0 python scripts/eval/brain_varnet_scaling_eval.py --seed 43 --split train
python scripts/eval/brain_varnet_scaling_summary.py
python scripts/plots/brain_varnet_scaling_trainer_val.py --seed 42 --out reports/brain_varnet_scaling/figures/trainer_val
python scripts/plots/brain_varnet_scaling_trainer_val.py --seed 43 --out reports/brain_varnet_scaling/figures/trainer_val_s43
CUDA_VISIBLE_DEVICES=0 python scripts/diagnostics/brain_trainval_data_audit.py
