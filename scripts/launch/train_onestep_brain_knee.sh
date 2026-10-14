#!/usr/bin/env bash
# One-step 2D sampler (conv_2d) trained end-to-end with a VarNet reconstructor on
# balanced brain T2 + knee data, with a widened sampler U-Net (chans 128 vs default 64).
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

mkdir -p checkpoints
python train.py --model conv \
    recon=varnet \
    datasets=brain_knee_balanced \
    training.num_step=1 \
    sampler.chans=128 \
    sampler.depth=4 \
    sampler.image_input=true \
    output.out_dir=checkpoints/onestep_2d_varnet_brain_knee_samp128 \
    "$@"
