#!/usr/bin/env bash
# Stage 2 of the static-VD VarNet learning-rate sweep: 50-epoch runs comparing schedulers at
# the lrs stage 1 left standing. The last run is the control the R=8 baselines used
# (lr 3e-4, step at epoch 40).
#
# One worker per GPU in GPUS; each takes the next unclaimed run once its GPU is idle, so a GPU
# that is busy with someone else's job just contributes no runs until it frees up.
# Run with: nohup scripts/launch/queue_static_vd_lr_stage2.sh &
set -euo pipefail

cd "$(dirname "$0")/../.."

GPUS="${GPUS:-0 2}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"  # idle GPUs here carry up to ~1.2 GB of another user's server
POLL_SECONDS=30
CLAIM_DIR=checkpoints/lr_sweep/stage2_claims
RUNS=("0.001 warmup_cosine" "0.003 warmup_cosine" "0.001 cosine" "0.0003 step")

gpu_used_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"
}

wait_for_idle_gpu() {
    until [[ "$(gpu_used_mib "$1")" -le "$MAX_USED_MIB" ]]; do sleep "$POLL_SECONDS"; done
}

wait_for_busy_gpu() {  # data loading leaves the GPU idle for minutes after launch
    until [[ "$(gpu_used_mib "$1")" -gt "$MAX_USED_MIB" ]]; do sleep "$POLL_SECONDS"; done
}

claim_next_run() {
    local run
    for run in "${RUNS[@]}"; do
        if mkdir "$CLAIM_DIR/${run// /_}" 2>/dev/null; then
            echo "$run"
            return 0
        fi
    done
    return 1
}

work_on_gpu() {
    local gpu="$1" run lr sched
    while true; do
        wait_for_idle_gpu "$gpu"
        run="$(claim_next_run)" || return 0
        read -r lr sched <<< "$run"
        echo "$(date '+%F %T') GPU $gpu launching lr=$lr sched=$sched"
        GPU="$gpu" LR="$lr" SCHED="$sched" EPOCHS=50 scripts/launch/train_static_vd_varnet_r8_brain_knee.sh datasets.num_workers=8
        wait_for_busy_gpu "$gpu"
        wait_for_idle_gpu "$gpu"
    done
}

mkdir -p "$CLAIM_DIR"
for gpu in $GPUS; do
    work_on_gpu "$gpu" &
    sleep "$POLL_SECONDS"  # stagger so the workers don't race for the first claim
done
wait
echo "$(date '+%F %T') stage 2 done"
