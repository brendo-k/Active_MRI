#!/usr/bin/env bash
# Retest labels (seed 1, every 4th labelled val slice) for the noise ceiling, then the go/no-go once all labels exist.
source "$(dirname "$0")/common.sh"
until [ -f $LABELS/val_0_640_s5.pt ]; do sleep 120; done
mkdir -p ${LABELS}_retest && cp $LABELS/prior.pt ${LABELS}_retest/
CUDA_VISIBLE_DEVICES=$(wait_gpu) python scripts/diagnostics/oracle_labels_r8.py --split val --count 32 --stride 20 --seed 1 \
    --steps 120 --draws 2 --chunk 8 --lr 0.05 --out-dir ${LABELS}_retest > analysis/label_gonogo/val_retest.log 2>&1
until [ -f $LABELS/train_0_600_s1.pt ] && [ -f $LABELS/train_600_1200_s1.pt ]; do sleep 120; done
CUDA_VISIBLE_DEVICES=$(wait_gpu) python scripts/diagnostics/label_predictor_gonogo_r8.py > analysis/label_gonogo/gonogo.log 2>&1
echo "gonogo done $(date)" >> analysis/label_gonogo/launcher.log
