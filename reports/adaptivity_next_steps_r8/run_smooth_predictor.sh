#!/usr/bin/env bash
# 2026-10-17, GPU 2: can the smooth part of each slice's oracle (worth +0.05-0.07 dB at sigma 4-8 in density space)
# be predicted from the calibration centre? Ridge / kNN predictors, static and volume-level controls.
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-2}
echo "start smooth_predictor $(date)" >> $LOG/launcher.log
python scripts/diagnostics/smooth_label_predictor_r8.py > $LOG/smooth_predictor.log 2>&1
rc=$?; echo "done smooth_predictor $(date) rc=$rc" >> $LOG/launcher.log
