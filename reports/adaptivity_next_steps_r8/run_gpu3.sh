#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-3}
sampler_run anattable_joint_R8 $LOUPE sampler.adapt_grid=0 sampler.anatomy_conditioning=true
sampler_run adapt16_frozen-loupe-recon_R8 $LOUPE sampler.adapt_grid=16 sampler.freeze_recon=true
