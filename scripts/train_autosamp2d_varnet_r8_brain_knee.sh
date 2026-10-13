#!/usr/bin/env bash
# 2D AutoSamp sampler (autosamp_2d: learned k-space positions snapped to the Cartesian
# grid) with a VarNet reconstructor at R=8 on balanced brain T2 + knee. The like-for-like
# counterpart of train_loupe2d_varnet_r8_brain_knee.sh and
# train_reinforce2d_varnet_r8_brain_knee.sh: same data, acceleration, reconstructor,
# batch size, epochs and reconstructor learning rate; only the sampler differs.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

GPU="${GPU:-0}"
OUT="${OUT:-checkpoints/autosamp2d_varnet_r8_brain_knee}"
BATCH_SIZE="${BATCH_SIZE:-16}"

source scripts/_launch.sh

mkdir -p "$(dirname "$OUT")"
launch_detached "$OUT.log" env CUDA_VISIBLE_DEVICES="$GPU" python scripts/train_sequential.py \
    recon=varnet \
    datasets=brain_knee_balanced \
    'sampler.accelerations=[8.0]' \
    sampler.sampler_type=autosamp_2d \
    training.batch_size="$BATCH_SIZE" \
    output.out_dir="$OUT" "$@"
