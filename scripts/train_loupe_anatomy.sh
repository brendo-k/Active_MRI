#!/usr/bin/env bash
# Separate zero-step LOUPE 2D samplers (VarNet recon, R=4) for brain and knee,
# 3 seeds each, to test (1) seed stability and (2) brain-vs-knee differences.
# Analyze afterwards with scripts/analyze_loupe_anatomy.py.
#
# Launched detached (setsid) so it outlives the shell. Queues run sequentially
# per GPU; override the layout with QUEUES="GPU:anat:seed,anat:seed GPU:..." .
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

ROOT="${ROOT:-$(pwd)/runs/loupe_anatomy}"
EPOCHS="${EPOCHS:-30}"
LR_STEP="${LR_STEP:-25}"
LR="${LR:-1e-3}"
WORKERS="${WORKERS:-6}"
QUEUES="${QUEUES:-0:brain:0,knee:2 1:brain:1 2:brain:2 3:knee:0,knee:1}"

mkdir -p "$ROOT/logs"

run_queue() {
    local gpu="$1"; shift
    for job in "$@"; do
        local anat="${job%%:*}" seed="${job##*:}"
        CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/train_sequential.py \
            recon=varnet \
            datasets="$anat" \
            training.num_step=0 \
            sampler.sampler_type=loupe_2d \
            training.seed="$seed" \
            training.num_epochs="$EPOCHS" \
            training.lr_step_size="$LR_STEP" \
            training.lr="$LR" \
            datasets.num_workers="$WORKERS" \
            output.out_dir="$ROOT/$anat/seed_$seed" \
            > "$ROOT/logs/${anat}_s${seed}.log" 2>&1
    done
}

for queue in $QUEUES; do
    gpu="${queue%%:*}"
    IFS=',' read -ra jobs <<< "${queue#*:}"
    setsid nohup bash -c "$(declare -f run_queue); ROOT='$ROOT' EPOCHS=$EPOCHS LR_STEP=$LR_STEP LR=$LR WORKERS=$WORKERS run_queue $gpu ${jobs[*]}" \
        > "$ROOT/logs/queue_gpu${gpu}.log" 2>&1 &
    echo "GPU $gpu queue [${jobs[*]}] PID $!"
done
