#!/usr/bin/env bash
# Matched short LOUPE 2D + VarNet runs (R=8, brain only) that differ only in the VarNet DC
# form: torch.where (no mask gradient through DC) vs mask-multiply with the full k-space as
# reference (mask gradient g * (k_pred - k_full) at unobserved locations). Forward passes
# are identical for a binary mask, so any difference comes from the mask gradient.
#
#   EPOCHS=30 GPU_WHERE=0 GPU_MULTIPLY=2 scripts/train_loupe_dc_mode_compare.sh
#   MODES=multiply GPU_MULTIPLY=1 scripts/train_loupe_dc_mode_compare.sh   # relaunch one arm
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

GPU_WHERE="${GPU_WHERE:-0}"
GPU_MULTIPLY="${GPU_MULTIPLY:-2}"
EPOCHS="${EPOCHS:-30}"
LR_STEP="${LR_STEP:-25}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-checkpoints/dc_mode_compare}"
MODES="${MODES:-where multiply}"

source scripts/_launch.sh
mkdir -p "$OUT_ROOT"

for mode in $MODES; do
    gpu_var="GPU_${mode^^}"
    out="$OUT_ROOT/loupe2d_varnet_r8_brainonly_dc${mode}_${EPOCHS}ep_seed${SEED}"
    launch_detached "$out.log" env CUDA_VISIBLE_DEVICES="${!gpu_var}" python scripts/train_sequential.py \
        recon=varnet \
        recon.dc_mode="$mode" \
        datasets=brain_knee_balanced \
        'datasets.train_anatomies=[brain]' \
        'sampler.accelerations=[8.0]' \
        sampler.sampler_type=loupe_2d \
        training.batch_size=16 \
        training.num_epochs="$EPOCHS" \
        training.lr_step_size="$LR_STEP" \
        training.seed="$SEED" \
        output.out_dir="$out" "$@"
done
