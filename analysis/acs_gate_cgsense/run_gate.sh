#!/usr/bin/env bash
# Stage 1 of the ACS mask gate (scripts/diagnostics/acs_mask_gate_cgsense_r8.py): waits for the 24 x 24 ecalib
# CG-SENSE population run to finish, then runs three shards (brain halves, knee) and the summary.
# Peak about 4 GB per shard (measured on the epoch-1 smoke test).
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

RUN_LOG=checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c24.log
until grep -q "Epoch = \[  50/  50\]" "$RUN_LOG"; do
    if grep -q "Traceback" "$RUN_LOG"; then echo "population run crashed"; exit 1; fi
    sleep 60
done
sleep 120

shard() {
    local gpu=$1; shift
    CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/acs_mask_gate_cgsense_r8.py "$@"
    local rc=$?
    echo "shard $* rc=$rc $(date)"
}
shard 3 --anatomy brain --volumes 0:12 > analysis/acs_gate_cgsense/brain_0.log 2>&1 &
shard 1 --anatomy brain --volumes 12: > analysis/acs_gate_cgsense/brain_1.log 2>&1 &
shard 3 --anatomy knee > analysis/acs_gate_cgsense/knee.log 2>&1 &
wait
python scripts/diagnostics/acs_mask_gate_cgsense_r8.py --summarize > analysis/acs_gate_cgsense/summary.log 2>&1
echo "gate done rc=$? $(date)"
