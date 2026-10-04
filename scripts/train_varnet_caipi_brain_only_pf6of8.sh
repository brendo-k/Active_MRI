#!/usr/bin/env bash
# Partial-Fourier test: same as the brain-only CAIPI baseline (wandb db7fsabe, lr 1e-3, no
# schedule, wd 1e-4, 50 epochs) but the rightmost 2/8 of columns are never sampled.
# If the VarNet recovers the missing phase/half-space, its val NMSE should stay near the baseline's.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python scripts/train_sequential.py \
    recon=varnet \
    datasets=brain_knee_balanced \
    'datasets.train_anatomies=[brain]' \
    training.num_step=1 \
    sampler.sampler_type=caipi_2d \
    sampler.caipi_keep_fraction=0.75 \
    training.lr=0.001 \
    training.num_epochs=50 \
    training.lr_step_size=1000 \
    training.weight_decay=1e-4 \
    output.out_dir=checkpoints/varnet_caipi2d_r4_brain_only_pf6of8 \
    "$@"
