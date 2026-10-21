#!/usr/bin/env bash
# The static and anatomy-table flagged arms ran out of memory on GPUs 1 and 3, which other runs had filled
# (2026-10-17 01:16). Rerun them one after the other on GPU 2, then the flagged readout.
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-2}
FLAGS="recon.adaptive_input=true recon.dc_weight_map=true"
BASE=$RUNS/loupe2d_flags_R8/last_model.pt
for name in static_joint_flags_R8 anattable_joint_flags_R8; do mv $RUNS/$name $RUNS/${name}_oom; done
INIT_MASK=$BASE sampler_run static_joint_flags_R8 $BASE sampler.adapt_grid=0 $FLAGS
INIT_MASK=$BASE sampler_run anattable_joint_flags_R8 $BASE sampler.adapt_grid=0 sampler.anatomy_conditioning=true $FLAGS
echo "start eval_flags_retry $(date)" >> $LOG/launcher.log
python scripts/eval/eval_adaptive_samplers_r8.py --tag flags --runs loupe_flags_original static_joint_flags_R8 \
  adapt16_joint_flags_R8 anattable_joint_flags_R8 > $LOG/eval_flags.log 2>&1
rc=$?; echo "done eval_flags_retry $(date) rc=$rc" >> $LOG/launcher.log
