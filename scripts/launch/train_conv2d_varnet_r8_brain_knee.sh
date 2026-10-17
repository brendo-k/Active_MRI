#!/usr/bin/env bash
# One-step 2D conv sampler (conv_2d, straight-through) with a VarNet reconstructor at R=8
# on balanced brain T2 + knee, with the mask-sigmoid and STE gradient floors on. Matches
# the LOUPE floor run (checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05) in
# data, acceleration, reconstructor, batch size, epochs, seed and floors; only the sampler differs.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

GPU="${GPU:-0}"
GRAD_FLOOR="${GRAD_FLOOR:-0.05}"
OUT="${OUT:-checkpoints/conv2d_varnet_r8_brain_knee_floor${GRAD_FLOOR}}"
BATCH_SIZE="${BATCH_SIZE:-16}"

source scripts/launch/_launch.sh

mkdir -p "$(dirname "$OUT")"
launch_detached "$OUT.log" env CUDA_VISIBLE_DEVICES="$GPU" python train.py --model conv \
    recon=varnet \
    datasets=brain_knee_balanced \
    training.num_step=1 \
    'sampler.accelerations=[8.0]' \
    sampler.mask_grad_floor="$GRAD_FLOOR" \
    sampler.ste_grad_floor="$GRAD_FLOOR" \
    training.batch_size="$BATCH_SIZE" \
    training.seed=42 \
    output.out_dir="$OUT" "$@"
