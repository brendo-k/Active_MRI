#!/usr/bin/env bash
# Flagged-VarNet readout once its three arms have finished.
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-3}
until grep -q "done static_joint_flags_R8" $LOG/launcher.log && grep -q "done adapt16_joint_flags_R8" $LOG/launcher.log \
      && grep -q "done anattable_joint_flags_R8" $LOG/launcher.log; do sleep 60; done
echo "start eval_flags $(date)" >> $LOG/launcher.log
python scripts/eval/eval_adaptive_samplers_r8.py --tag flags --runs loupe_flags_original static_joint_flags_R8 \
  adapt16_joint_flags_R8 anattable_joint_flags_R8 > $LOG/eval_flags.log 2>&1
echo "done eval_flags $(date) rc=$?" >> $LOG/launcher.log
