#!/usr/bin/env bash
# 2026-10-17, GPU 1: (1) the oracle-blur test redone with density-space blur (the logit-space version moves budget
# between radial rings); (2) one sampler map per protocol (brain / knee-PD / knee-PDFS), the anatomy-table recipe with
# three classes, then its readout against the static control and the two-class anatomy table.
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
echo "start oracle_blur_density $(date)" >> $LOG/launcher.log
python scripts/diagnostics/oracle_blur_psnr_r8.py --space density --out analysis/label_gonogo/oracle_blur_density.json \
  > $LOG/oracle_blur_density.log 2>&1
rc=$?; echo "done oracle_blur_density $(date) rc=$rc" >> $LOG/launcher.log
sampler_run prototable_joint_R8 $LOUPE sampler.adapt_grid=0 sampler.anatomy_conditioning=true \
  'datasets.condition_classes=[brain,knee-PD,knee-PDFS]'
echo "start eval_protocol $(date)" >> $LOG/launcher.log
python scripts/eval/eval_adaptive_samplers_r8.py --tag protocol --runs static_joint_R8 anattable_joint_R8 \
  prototable_joint_R8 > $LOG/eval_protocol.log 2>&1
rc=$?; echo "done eval_protocol $(date) rc=$rc" >> $LOG/launcher.log
