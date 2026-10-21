#!/usr/bin/env bash
# After the generalist reconstructor: the adaptive head and its static control against it, frozen.
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
until grep -q "done generalist" $LOG/launcher.log; do sleep 60; done
sampler_run adapt16_frozen-generalist_R8 $GENERALIST sampler.adapt_grid=16 sampler.freeze_recon=true
sampler_run static_frozen-generalist_R8 $GENERALIST sampler.adapt_grid=0 sampler.freeze_recon=true
