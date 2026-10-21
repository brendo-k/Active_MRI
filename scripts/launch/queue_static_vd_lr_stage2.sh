#!/usr/bin/env bash
# Stage 2 of the static-VD VarNet learning-rate sweep: 50-epoch runs comparing schedulers at
# the lrs stage 1 left standing, in priority order. The first run is the control the current
# R=8 baselines use (lr 1e-3, step at epoch 40); the last is the older lr 3e-4 baseline.
#
# One worker per GPU in GPUS; each takes the next unclaimed run once its GPU is idle, so a GPU
# that is busy with someone else's job just contributes no runs until it frees up.
#
# A run whose job exits before its last epoch (OOM when another job fills the GPU, a kill, a
# crash) is requeued from scratch: its out dir and log are renamed *_crashed<attempt> and its
# claim is released, so the next idle worker relaunches it. After MAX_ATTEMPTS the claim is kept
# with a `gave_up` marker. Each claim dir holds the job's PID in `pid`.
# Run with: nohup scripts/launch/queue_static_vd_lr_stage2.sh &
set -euo pipefail
shopt -s nullglob

cd "$(dirname "$0")/../.."

GPUS="${GPUS:-0 2}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"  # idle GPUs here carry up to ~1.2 GB of another user's server
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
POLL_SECONDS=30
EPOCHS=50
SEED=42
CLAIM_DIR=checkpoints/lr_sweep/stage2_claims
LOCK_DIR="$CLAIM_DIR.lock"
RUNS=("0.001 step" "0.001 cosine" "0.001 warmup_cosine" "0.003 warmup_cosine" "0.0003 step")

gpu_used_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" || echo 999999
}

wait_for_idle_gpu() {
    until [[ "$(gpu_used_mib "$1")" -le "$MAX_USED_MIB" ]]; do sleep "$POLL_SECONDS"; done
}

run_out() {  # the train script's default OUT for this lr and scheduler
    echo "checkpoints/lr_sweep/static_vd_lr${1}_${2}_${EPOCHS}ep_seed${SEED}"
}

job_alive() {  # claim dir -> true while its job is launching (no pid yet) or running
    local claim="$1" pid
    [[ -f "$claim/pid" ]] || return 0
    pid=$(<"$claim/pid")
    [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "train.py"  # guards against PID reuse
}

any_job_alive() {
    local claim
    for claim in "$CLAIM_DIR"/*/; do
        claim=${claim%/}
        [[ -e "$claim/gave_up" ]] && continue
        job_alive "$claim" && return 0
    done
    return 1
}

lock() { until mkdir "$LOCK_DIR" 2>/dev/null; do sleep 1; done; }
unlock() { rmdir "$LOCK_DIR"; }

release_crashed_runs() {  # caller holds the lock
    local claim name out attempt crashed_logs
    for claim in "$CLAIM_DIR"/*/; do
        claim=${claim%/}
        name=${claim##*/}
        [[ -e "$claim/gave_up" ]] && continue
        job_alive "$claim" && continue
        out=$(run_out "${name%%_*}" "${name#*_}")
        grep -qE "Epoch = \[ *${EPOCHS}/ *${EPOCHS}\]" "$out.log" 2>/dev/null && continue
        crashed_logs=("$out"_crashed*.log)
        attempt=$(( ${#crashed_logs[@]} + 1 ))
        [[ -e "$out" ]] && mv "$out" "${out}_crashed${attempt}"
        [[ -e "$out.log" ]] && mv "$out.log" "${out}_crashed${attempt}.log"
        if (( attempt >= MAX_ATTEMPTS )); then
            touch "$claim/gave_up"
            echo "$(date '+%F %T') $name crashed on attempt $attempt of $MAX_ATTEMPTS; giving up"
        else
            rm -rf "$claim"
            echo "$(date '+%F %T') $name crashed on attempt $attempt of $MAX_ATTEMPTS; requeued"
        fi
    done
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

launch_run() {  # gpu, lr, scheduler, claim dir; records the job's PID (or `none`) in the claim
    local gpu="$1" lr="$2" sched="$3" claim="$4" launch pid
    echo "$(date '+%F %T') GPU $gpu launching lr=$lr sched=$sched"
    launch=$(GPU="$gpu" LR="$lr" SCHED="$sched" EPOCHS="$EPOCHS" SEED="$SEED" \
        scripts/launch/train_static_vd_varnet_r8_brain_knee.sh datasets.num_workers=8 2>&1) || true
    echo "$launch"
    pid=$(sed -n 's/^PID: //p' <<< "$launch")
    echo "${pid:-none}" > "$claim/pid"
}

work_on_gpu() {
    local gpu="$1" run lr sched claim
    while true; do
        wait_for_idle_gpu "$gpu"
        lock
        release_crashed_runs
        run="$(claim_next_run)" || run=""
        unlock
        if [[ -z "$run" ]]; then
            any_job_alive || return 0
            sleep "$POLL_SECONDS"
            continue
        fi
        read -r lr sched <<< "$run"
        claim="$CLAIM_DIR/${run// /_}"
        launch_run "$gpu" "$lr" "$sched" "$claim"
        while job_alive "$claim"; do sleep "$POLL_SECONDS"; done
    done
}

main() {
    mkdir -p "$CLAIM_DIR"
    for gpu in $GPUS; do
        work_on_gpu "$gpu" &
        sleep "$POLL_SECONDS"  # stagger so the workers don't race for the first claim
    done
    wait
    echo "$(date '+%F %T') stage 2 done"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main
fi
