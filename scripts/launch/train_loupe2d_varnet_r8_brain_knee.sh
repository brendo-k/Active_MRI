#!/usr/bin/env bash
# 2D LOUPE sampler (loupe_2d, straight-through) with a VarNet reconstructor at R=8
# on balanced brain T2 + knee. The like-for-like baseline for
# train_reinforce2d_varnet_r8_brain_knee.sh: same data, acceleration, reconstructor,
# batch size, epochs and learning rate; only the mask-training mode differs.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

GPU="${GPU:-0}"
OUT="${OUT:-checkpoints/loupe2d_varnet_r8_brain_knee}"
BATCH_SIZE="${BATCH_SIZE:-16}"

source scripts/launch/_launch.sh

mkdir -p "$(dirname "$OUT")"
launch_detached "$OUT.log" env CUDA_VISIBLE_DEVICES="$GPU" python train.py --model loupe \
    recon=varnet \
    datasets=brain_knee_balanced \
    'sampler.accelerations=[8.0]' \
    training.batch_size="$BATCH_SIZE" \
    output.out_dir="$OUT" "$@"
