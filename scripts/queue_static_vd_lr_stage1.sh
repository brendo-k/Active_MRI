#!/usr/bin/env bash
# Stage 1 of the static-VD VarNet learning-rate sweep: constant-lr 10-epoch runs, one after
# another on one GPU (a batch-16 VarNet run saturates the card). Short horizons favour high
# lrs, so this only finds the stable, fast range; schedulers are compared at full length in
# stage 2. The lr=1e-3 run is launched by hand, so this queue only adds the other two.
# Run with: nohup scripts/queue_static_vd_lr_stage1.sh &
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
MAX_USED_MIB=1000
POLL_SECONDS=30

wait_for_idle_gpu() {
    until [[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")" -le "$MAX_USED_MIB" ]]; do
        sleep "$POLL_SECONDS"
    done
}

for lr in 0.0003 0.003; do
    wait_for_idle_gpu
    echo "$(date '+%F %T') launching lr=$lr on GPU $GPU"
    GPU="$GPU" LR="$lr" SCHED=constant EPOCHS=10 scripts/train_static_vd_varnet_r8_brain_knee.sh datasets.num_workers=8
    sleep 120  # let the new job allocate before the next GPU check
done
echo "$(date '+%F %T') stage 1 queued"
