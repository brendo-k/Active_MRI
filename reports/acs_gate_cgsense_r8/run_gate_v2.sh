#!/usr/bin/env bash
# Rerun of the ACS gate (scripts/diagnostics/acs_mask_gate_cgsense_r8.py) with fitting settings fixed by the
# in-sample check (scripts/diagnostics/acs_gate_fit_check_r8.py, reports/acs_gate_cgsense_r8/fit_check_*_lr.json).
# The first run (free lr 0.05, grid lr 0.02, 300 steps) failed it: grid_in lost to the population mask on all 10
# knee volumes. Here free lr 0.005, grid lr 0.002, 600 steps: in-sample fits at or above base on 3 knee and 2 brain
# volumes. Peak about 4 GB per shard; GPU 3 is shared with another session's 20 GB run, GPU 0 with the FIRE server.
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

shard() {
    local gpu=$1 log=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/acs_mask_gate_cgsense_r8.py --tag v2 --mask-lr 0.005 \
        --grid-lr 0.002 --steps 600 "$@" > "reports/acs_gate_cgsense_r8/$log.log" 2>&1
    local rc=$?
    echo "shard $log rc=$rc $(date)"
}
shard 3 brain_v2_0 --anatomy brain --volumes 0:8 &
shard 3 brain_v2_1 --anatomy brain --volumes 8:16 &
shard 3 brain_v2_2 --anatomy brain --volumes 16: &
shard 0 knee_v2 --anatomy knee &
wait
python scripts/diagnostics/acs_mask_gate_cgsense_r8.py --summarize --tag v2 > reports/acs_gate_cgsense_r8/summary_v2.log 2>&1
rc=$?
echo "gate v2 done rc=$rc $(date)"
