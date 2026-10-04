#!/usr/bin/env bash
# Same as train_varnet_caipi_brain_knee.sh, plus FiLM conditioning on each VarNet cascade
# bottleneck with one class per contrast: brain T2, knee PD, knee PDFS (fat sat).
# lr is pinned to the baseline's 1e-3 (configs/config.yaml has since moved to 3e-4).
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

mkdir -p checkpoints
python scripts/train_sequential.py \
    recon=varnet \
    recon.film_conditioning=true \
    datasets=brain_knee_balanced \
    'datasets.condition_classes=[brain,knee-PD,knee-PDFS]' \
    training.num_step=1 \
    sampler.sampler_type=caipi_2d \
    training.lr=0.001 \
    output.out_dir=checkpoints/varnet_caipi2d_r4_brain_knee_film \
    "$@"
