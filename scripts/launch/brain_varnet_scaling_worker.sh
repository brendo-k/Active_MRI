#!/usr/bin/env bash
# Copy of varnet_train_speed_worker.sh that waits for MIN_FREE_MIB free on its GPU instead of for an empty GPU
# (other users keep ~300 MB processes on every card).
# One GPU's worker for a training queue (written for reports/varnet_train_speed, reused by later studies).
#
# Jobs are lines of $QUEUE_DIR/jobs.txt:   <name>|<code snapshot dir>|<estimated minutes>|<hydra overrides...>
# A worker claims the first job without a claim directory (mkdir is atomic, so workers never run the same job), runs it
# from its code snapshot on its own GPU, and moves on. Lines can be appended while the workers run. A job whose
# estimate would end past $DEADLINE is left unclaimed: HH:MM (the next one, within 12 h), a date such as
# '2026-11-05 06:00', or none. A run that stops before its last epoch is rerun from scratch (no resume) up to
# MAX_ATTEMPTS times; its log is kept as <name>.attemptN.log. Every job line must spell out training.num_epochs=N:
# the done check looks for its last epoch, or for the trainer's "Early stop after epoch" line. Runs log to the wandb group $RUN_GROUP as $RUN_PREFIX<name>.
#
#   QUEUE_DIR=~/data/seqmri_checkpoints/varnet_train_speed GPU=0 nohup setsid <copy of this script> > worker0.log 2>&1 &
# Run a copy of this script that nothing edits (bash reads a script as it goes).
set -uo pipefail

QUEUE_DIR="${QUEUE_DIR:?set QUEUE_DIR}"
GPU="${GPU:?set GPU}"
DEADLINE="${DEADLINE:-06:50}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
RUN_GROUP="${RUN_GROUP:-varnet_train_speed}"
RUN_PREFIX="${RUN_PREFIX:-vts_}"
PY="${PY:-/home/bkadota/code/active_mri/SeqMRI/.venv/bin/python}"
GREP=/usr/bin/grep
JOBS="$QUEUE_DIR/jobs.txt"
CLAIMS="$QUEUE_DIR/claims"
mkdir -p "$CLAIMS"

stamp() { date '+%F %T'; }
deadline_epoch() {
    local today_deadline
    [[ "$DEADLINE" == none ]] && { echo 99999999999; return; }
    [[ "$DEADLINE" =~ ^[0-9]{1,2}:[0-9]{2}$ ]] || { date -d "$DEADLINE" +%s; return; }
    today_deadline=$(date -d "today $DEADLINE" +%s)
    if (( today_deadline < $(date +%s) - 43200 )); then date -d "tomorrow $DEADLINE" +%s; else echo "$today_deadline"; fi
}
DEADLINE_EPOCH=$(deadline_epoch) || exit 2

MIN_FREE_MIB="${MIN_FREE_MIB:-44000}"
gpu_lacks_room() { (( $(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits) < MIN_FREE_MIB )); }

run_job() {
    local name=$1 code=$2 overrides=$3 out log epochs rc attempt
    out="$QUEUE_DIR/$name"
    log="$QUEUE_DIR/$name.log"
    epochs=$(echo "$overrides" | $GREP -oE 'training\.num_epochs=[0-9]+' | cut -d= -f2)
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        while gpu_lacks_room; do echo "$(stamp) GPU $GPU has < $MIN_FREE_MIB MiB free, waiting"; sleep 60; done
        echo "$(stamp) start $name GPU $GPU attempt $attempt"
        (
            cd "$code" || exit 97
            export PYTHONPATH="$code" CUDA_VISIBLE_DEVICES="$GPU" WANDB_RUN_GROUP="$RUN_GROUP"
            set -f
            # shellcheck disable=SC2086
            exec "$PY" train.py --model sequential $overrides output.out_dir="$out" output.run_name="$RUN_PREFIX$name"
        ) > "$log" 2>&1
        rc=$?
        if $GREP -qE "Epoch = \[ *$epochs/ *$epochs\]|Early stop after epoch" "$log"; then
            echo "$(stamp) done $name (exit $rc)"
            echo "$rc" > "$CLAIMS/$name/done"
            return 0
        fi
        mv "$log" "$QUEUE_DIR/$name.attempt$attempt.log"
        echo "$(stamp) $name stopped early (exit $rc); log kept as $name.attempt$attempt.log"
        [[ -d "$out" ]] && mv "$out" "$out.attempt$attempt"
        sleep 30
    done
    echo "$(stamp) gave up on $name"
    touch "$CLAIMS/$name/gave_up"
    return 1
}

echo "$(stamp) worker GPU $GPU start, deadline $(date -d "@$DEADLINE_EPOCH" '+%F %T')"
while [[ ! -e "$QUEUE_DIR/STOP" ]]; do
    claimed=""
    while IFS='|' read -r name code minutes overrides; do
        [[ -z "$name" || "$name" == \#* ]] && continue
        [[ -d "$CLAIMS/$name" ]] && continue
        if (( $(date +%s) + minutes * 60 > DEADLINE_EPOCH )); then continue; fi
        if mkdir "$CLAIMS/$name" 2>/dev/null; then
            echo "$GPU" > "$CLAIMS/$name/gpu"
            claimed="$name|$code|$overrides"
            break
        fi
    done < "$JOBS"
    if [[ -z "$claimed" ]]; then
        (( $(date +%s) > DEADLINE_EPOCH )) && break
        sleep 60
        continue
    fi
    IFS='|' read -r name code overrides <<< "$claimed"
    run_job "$name" "$code" "$overrides"
done
echo "$(stamp) worker GPU $GPU exit"
