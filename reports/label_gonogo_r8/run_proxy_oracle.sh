#!/usr/bin/env bash
# Proxy-target oracle on GPU 0: labels fitted against the zero-filled 10/24/64 centre instead of the ground truth.
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=0
mkdir -p /tmp/bkadota_label_gonogo
python scripts/diagnostics/proxy_target_oracle_r8.py --centres 10 24 64 > /tmp/bkadota_label_gonogo/proxy_target_oracle.log 2>&1
echo "proxy oracle done $(date)" >> reports/label_gonogo_r8/launcher.log
