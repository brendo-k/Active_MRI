#!/usr/bin/env bash
# Adaptivity comparison (now including the clamp run) once the clamp run reaches epoch 50.
source "$(dirname "$0")/common.sh"
until grep -q "Epoch = \[  50/  50\]" checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_clamp10_leak0.05.log; do sleep 120; done
sleep 120
CUDA_VISIBLE_DEVICES=$(wait_gpu) python scripts/eval/compare_conv_adaptivity_r8.py > reports/conv_vs_loupe_r8/adaptivity.log 2>&1
echo "adaptivity done $(date)" >> analysis/label_gonogo/launcher.log
