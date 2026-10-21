#!/usr/bin/env bash
# Adaptive-head reruns after bounding the offset (tanh, +/-6) and giving the predictor lr 1e-4.
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-3}
sampler_run adapt16_joint_R8 $LOUPE sampler.adapt_grid=16
sampler_run adapt16_frozen-loupe-recon_R8 $LOUPE sampler.adapt_grid=16 sampler.freeze_recon=true
