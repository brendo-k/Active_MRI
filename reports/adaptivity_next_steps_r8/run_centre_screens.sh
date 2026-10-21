#!/usr/bin/env bash
# Can a larger calibration block predict the per-slice oracle masks? Go/no-go screens that differ only in the
# predictor's input: the 10x10 centre (today's sampler), a 24x24 centre, and the full k-space (ceiling).
source "$(dirname "$0")/common.sh"
OUT=reports/adaptivity_next_steps_r8
screen() { name=$1; gpu=$2; shift 2
  echo "start screen_$name $(date)" >> $LOG/launcher.log
  CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/label_predictor_gonogo_r8.py --screen --out $OUT/screen_$name.json "$@" > $LOG/screen_$name.log 2>&1
  echo "done screen_$name $(date) rc=$?" >> $LOG/launcher.log; }
(screen centre24 1 --predictor-input centre --centre-size 24; screen observed10 1 --predictor-input observed) &
screen full 3 --predictor-input full &
wait
