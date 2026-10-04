#!/usr/bin/env bash
# VarNet trained on a fixed 2D-CAIPI (2x2, shift 1) R=4 pattern on balanced brain T2 + knee data.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

mkdir -p checkpoints
python scripts/train_sequential.py \
    recon=varnet \
    datasets=brain_knee_balanced \
    training.num_step=1 \
    sampler.sampler_type=caipi_2d \
    output.out_dir=checkpoints/varnet_caipi2d_r4_brain_knee \
    "$@"
