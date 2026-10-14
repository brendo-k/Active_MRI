#!/usr/bin/env bash
# Brain-only control for train_varnet_caipi_brain_knee.sh: same balanced split and hyperparameters
# (lr pinned to the _samedata baseline's 1e-3), but trained on the brain slices alone.
# Validation still covers brain and knee, so knee metrics measure cross-anatomy transfer.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

mkdir -p checkpoints
python train.py --model sequential \
    recon=varnet \
    datasets=brain_knee_balanced \
    'datasets.train_anatomies=[brain]' \
    training.num_step=1 \
    sampler.sampler_type=caipi_2d \
    training.lr=0.001 \
    output.out_dir=checkpoints/varnet_caipi2d_r4_brain_only \
    "$@"
