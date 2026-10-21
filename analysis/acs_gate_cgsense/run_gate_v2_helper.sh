#!/usr/bin/env bash
# Helper shard for run_gate_v2.sh: once the knee shard is done, GPU 0 takes brain volumes from the end of the list.
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
until grep -q "shard knee_v2 rc=" analysis/acs_gate_cgsense/run_gate_v2.log; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 python scripts/diagnostics/acs_mask_gate_cgsense_r8.py --tag v2 --mask-lr 0.005 --grid-lr 0.002 \
    --steps 600 --anatomy brain --reverse > analysis/acs_gate_cgsense/brain_v2_helper.log 2>&1
rc=$?
echo "helper rc=$rc $(date)"
