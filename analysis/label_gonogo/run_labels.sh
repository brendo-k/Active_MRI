#!/usr/bin/env bash
# Oracle labels for the go/no-go test: GPU 0 runs train shard 0-600 then 128 val labels;
# GPU 1 runs train shard 600-1200 once the clamp run has released it.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
COMMON="--steps 120 --draws 2 --chunk 8 --lr 0.05"
LOG=analysis/label_gonogo
( CUDA_VISIBLE_DEVICES=0 python scripts/diagnostics/oracle_labels_r8.py --split train --start 0 --count 600 $COMMON > $LOG/train_0_600.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python scripts/diagnostics/oracle_labels_r8.py --split val --count 128 --stride 5 $COMMON > $LOG/val_128.log 2>&1 ) &
( sleep 600
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 20000 ]; do sleep 60; done
  echo "GPU 1 free at $(date)" >> $LOG/launcher.log
  CUDA_VISIBLE_DEVICES=1 python scripts/diagnostics/oracle_labels_r8.py --split train --start 600 --count 600 $COMMON > $LOG/train_600_1200.log 2>&1 ) &
wait
echo "labels done $(date)" >> $LOG/launcher.log
