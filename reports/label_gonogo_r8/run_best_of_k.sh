#!/usr/bin/env bash
# Best-of-K selection upper bound on GPU 0: K = 4/16/64 per-anatomy medoids of the training oracle labels.
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=0
mkdir -p /tmp/bkadota_label_gonogo
python scripts/diagnostics/best_of_k_oracle_r8.py --ks 4 16 64 > /tmp/bkadota_label_gonogo/best_of_k_oracle.log 2>&1
echo "best-of-k done $(date)" >> reports/label_gonogo_r8/launcher.log
