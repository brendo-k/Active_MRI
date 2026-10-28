#!/usr/bin/env bash
# Queue for scripts/launch/train_patient_specific_r8_local_v1_flair_knee.sh on two GPUs, CONV_GPU and LOUPE_GPU.
#   1. waits for $WAIT_LOG (if set) to report "all waves finished" and for both GPUs to be idle
#   2. trains the arms in $ARMS (default "$ADAPTIVE_ARM loupe") at once; an arm left out is waited for while its run is alive. A run that runs out of GPU memory is rerun with
#      recon.checkpoint_cascades=true (same numbers, less memory); any other stop is rerun from scratch once
#   3. scores the pair ($ADAPTIVE_ARM, default conv, against loupe) on CONV_GPU with scripts/eval/eval_patient_specific_r8.py
#      into $REPO/reports/patient_specific_r8_local_v1/seed$SEED (conv) or <adaptive arm>_seed$SEED (any other arm)
# Adaptive arms train on CONV_GPU, loupe on LOUPE_GPU.
# Run it from a code snapshot so later edits to the repo cannot reach the runs, reruns or eval:
#   REPO=<repo> nohup setsid <snapshot>/scripts/launch/queue_patient_specific_r8_local_v1_flair_knee.sh > <log> 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."
CODE="$(pwd)"
PY="${PY:-/data/yanglab2/labmembers2/bkadota/conda/envs/dl/bin/python}"
export PATH="$(dirname "$PY"):$PATH"
GREP=/usr/bin/grep
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
OUT_ROOT="${OUT_ROOT:-$HOME/data/seqmri_checkpoints/patient_specific_r8_local_v1/runs}"
REPO="${REPO:?set REPO to the repository whose reports folder gets the eval JSON}"
REPORTS="$REPO/${REPORTS:-reports/patient_specific_r8_local_v1}"
WAIT_LOG="${WAIT_LOG:-}"
CONV_GPU="${CONV_GPU:-0}"
LOUPE_GPU="${LOUPE_GPU:-1}"
ADAPTIVE_ARM="${ADAPTIVE_ARM:-conv}"
ARMS="${ARMS:-$ADAPTIVE_ARM loupe}"
if [[ $ADAPTIVE_ARM == conv ]]; then EVAL_NAME="seed$SEED"; else EVAL_NAME="${ADAPTIVE_ARM}_seed$SEED"; fi
LAST_EPOCH="Epoch = \[ *$EPOCHS/ *$EPOCHS\]"
mkdir -p "$REPORTS" "$OUT_ROOT"

stamp() { date '+%F %T'; }
run_dir() { echo "$OUT_ROOT/patient_specific_${1}_r8_flair_knee_${EPOCHS}ep_seed${SEED}"; }
gpu_busy() { [[ -n "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader)" ]]; }
waiting_on_log() { [[ -n "$WAIT_LOG" ]] && ! { [[ -f "$WAIT_LOG" ]] && $GREP -q 'all waves finished' "$WAIT_LOG"; }; }

arm_gpu() { if [[ $1 == loupe ]]; then echo "$LOUPE_GPU"; else echo "$CONV_GPU"; fi; }
any_arm_gpu_busy() { for arm in $ARMS; do gpu_busy "$(arm_gpu "$arm")" && return 0; done; return 1; }

echo "$(stamp) waiting for '${WAIT_LOG:-nothing}' and the GPUs of '$ARMS' to be idle"
while waiting_on_log || any_arm_gpu_busy; do sleep 120; done

train() {
    local arm=$1 gpu=$2 out extra=()
    out="$(run_dir "$arm")"
    if $GREP -qE "$LAST_EPOCH" "$out.log" 2>/dev/null; then
        echo "$(stamp) $arm already finished; not retrained"
        return 0
    fi
    for attempt in 1 2 3; do
        echo "$(stamp) start $arm GPU $gpu attempt $attempt ${extra[*]}"
        FOREGROUND=1 GPU="$gpu" ARM="$arm" SEED="$SEED" EPOCHS="$EPOCHS" OUT_ROOT="$OUT_ROOT" \
            scripts/launch/train_patient_specific_r8_local_v1_flair_knee.sh "${extra[@]}"
        if $GREP -qE "$LAST_EPOCH" "$out.log"; then
            echo "$(stamp) done $arm"
            return 0
        fi
        mv "$out.log" "$out.attempt$attempt.log"
        if $GREP -q 'out of memory' "$out.attempt$attempt.log"; then
            echo "$(stamp) $arm ran out of GPU memory; rerunning with recon.checkpoint_cascades=true"
            extra=(recon.checkpoint_cascades=true)
        else
            echo "$(stamp) $arm stopped early; log kept as $out.attempt$attempt.log"
            [[ $attempt -ge 2 ]] && return 1
        fi
        sleep 120
    done
    return 1
}

for arm in $ARMS; do
    train "$arm" "$(arm_gpu "$arm")" &
done
wait

# An arm left out of ARMS is trained elsewhere (e.g. an earlier queue that still runs it); score once it finishes.
finished() { $GREP -qE "$LAST_EPOCH" "$(run_dir "$1").log" 2>/dev/null; }
running() { [[ -n "$(ps -eo args | $GREP -F "output.out_dir=$(run_dir "$1")" | $GREP -v grep)" ]]; }
for arm in $ADAPTIVE_ARM loupe; do
    while ! finished "$arm" && running "$arm"; do sleep 300; done
done

if finished "$ADAPTIVE_ARM" && finished loupe; then
    for attempt in 1 2 3; do
        (cd "$REPO" && CUDA_VISIBLE_DEVICES="$CONV_GPU" "$PY" "$CODE/scripts/eval/eval_patient_specific_r8.py" \
            --adaptive "$(run_dir "$ADAPTIVE_ARM")" --control "$(run_dir loupe)" --out "$REPORTS/$EVAL_NAME") \
            > "$REPORTS/patient_specific_${EVAL_NAME}_stdout.txt" 2>&1
        rc=$?
        echo "$(stamp) eval attempt $attempt exit $rc"
        [[ $rc == 0 ]] && break
        sleep 300
    done
else
    echo "$(stamp) a run did not finish; not scored"
fi
echo "$(stamp) queue finished"
