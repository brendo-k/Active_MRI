#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-2}
sampler_run adapt16_joint_R8 $LOUPE sampler.adapt_grid=16
sampler_run static_joint_R8 $LOUPE sampler.adapt_grid=0
