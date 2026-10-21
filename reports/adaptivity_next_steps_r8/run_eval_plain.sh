#!/usr/bin/env bash
# Plain-VarNet readout once every plain arm has finished (the bounded reruns log a second "done" line).
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-3}
until [ $(grep -c "done adapt16_frozen-loupe-recon_R8" $LOG/launcher.log) -ge 2 ] && grep -q "done static_frozen-generalist_R8" $LOG/launcher.log; do sleep 60; done
echo "start eval_plain $(date)" >> $LOG/launcher.log
python scripts/eval/eval_adaptive_samplers_r8.py --tag plain --runs loupe_original static_joint_R8 adapt16_joint_R8 \
  anattable_joint_R8 adapt16_frozen-loupe-recon_R8 adapt16_frozen-generalist_R8 static_frozen-generalist_R8 > $LOG/eval_plain.log 2>&1
echo "done eval_plain $(date) rc=$?" >> $LOG/launcher.log
