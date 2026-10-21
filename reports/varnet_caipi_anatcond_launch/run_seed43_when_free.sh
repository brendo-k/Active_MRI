#!/usr/bin/env bash
# Second seed of the PromptMR+ CAIPI anatomy-conditioning pair: each arm starts on the next empty GPU among 1 and 3
# (0 and 2 hold seed 42), checking free memory when it launches.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

ALLOWED_GPUS='^[13]$'
EMPTY_MIB=1000
LOG=reports/varnet_caipi_anatcond_launch/launcher.log

empty_gpu() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F', ' -v allowed="$ALLOWED_GPUS" -v empty="$EMPTY_MIB" '$1 ~ allowed && $2 < empty {print $1; exit}'
}

wait_for_gpu() {
    local gpu=""
    until [ -n "$gpu" ]; do
        sleep 120
        gpu=$(empty_gpu)
    done
    sleep 60  # let a job that is still loading claim its memory before re-checking
    gpu=$(empty_gpu)
    if [ -z "$gpu" ]; then wait_for_gpu; return; fi
    echo "$gpu"
}

for arm in cond blind; do
    echo "seed 43 $arm: waiting for an empty GPU in 1/3, $(date)" >> "$LOG"
    gpu=$(wait_for_gpu)
    echo "seed 43 $arm: launching on GPU $gpu, $(date)" >> "$LOG"
    ARM=$arm GPU=$gpu SEED=43 STARTUP_CHECK_SECONDS=90 \
        scripts/launch/train_varnet_caipi_promptmr_anatcond.sh >> "$LOG" 2>&1
done
