#!/usr/bin/env bash
# Runs the arms of train_unet_speed_varnet2_r8.sh (CASCADES cascades, default 2), seeds 42 then 43, one run per GPU:
# each GPU in GPUS takes one arm (in ARMS order), so the seed-42 wave finishes as a complete comparison before seed 43
# starts.
#
# Each worker waits for its GPU to be idle before a run. A run that ends before its last epoch (OOM, kill, crash) is
# rerun from scratch, its out dir and log renamed *_crashed<attempt>; after MAX_ATTEMPTS it moves on.
# Run with: nohup scripts/launch/queue_unet_speed_varnet2_r8.sh > <log> 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."

GPUS=(${GPUS:-0 1 2 3})
ARMS=(${ARMS:-base bf16 sens64})
SEEDS=(${SEEDS:-42 43})
CASCADES="${CASCADES:-2}"
EPOCHS=50
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
OUT_ROOT="${OUT_ROOT:-$HOME/data/seqmri_checkpoints/unet_speed/runs}"

stamp() { date '+%F %T'; }

wait_for_idle_gpu() {
    local used
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 999999)
        [[ "$used" -le "$MAX_USED_MIB" ]] && return
        sleep 30
    done
}

worker() {
    local gpu="$1" arm="$2" seed attempt out
    for seed in "${SEEDS[@]}"; do
        out="$OUT_ROOT/varnet${CASCADES}_r8_${arm}_seed${seed}"
        for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
            wait_for_idle_gpu "$gpu"
            echo "$(stamp) gpu$gpu start $arm seed$seed attempt $attempt"
            FOREGROUND=1 ARM="$arm" GPU="$gpu" SEED="$seed" CASCADES="$CASCADES" OUT_ROOT="$OUT_ROOT" \
                scripts/launch/train_unet_speed_varnet2_r8.sh
            if grep -qE "Epoch = \[ *${EPOCHS}/ *${EPOCHS}\]" "$out.log" 2>/dev/null; then
                echo "$(stamp) gpu$gpu done $arm seed$seed"
                break
            fi
            echo "$(stamp) gpu$gpu CRASHED $arm seed$seed attempt $attempt: $(tail -n 1 "$out.log" 2>/dev/null)"
            mv "$out" "${out}_crashed$attempt" 2>/dev/null
            mv "$out.log" "${out}_crashed$attempt.log" 2>/dev/null
        done
    done
}

mkdir -p "$OUT_ROOT"
for i in "${!ARMS[@]}"; do
    worker "${GPUS[$i]}" "${ARMS[$i]}" &
done
wait
echo "$(stamp) queue finished"
