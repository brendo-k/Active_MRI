#!/usr/bin/env bash
# Flagged phase: anatomy-table positive control, after GPU 1's plain runs and the flagged base.
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
FLAGS="recon.adaptive_input=true recon.dc_weight_map=true"
until grep -q "done static_frozen-generalist_R8" $LOG/launcher.log && grep -q "done loupe2d_flags_R8" $LOG/launcher.log; do sleep 60; done
INIT_MASK=$RUNS/loupe2d_flags_R8/last_model.pt sampler_run anattable_joint_flags_R8 $RUNS/loupe2d_flags_R8/last_model.pt sampler.adapt_grid=0 sampler.anatomy_conditioning=true $FLAGS
