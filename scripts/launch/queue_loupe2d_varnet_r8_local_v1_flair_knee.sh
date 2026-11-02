#!/usr/bin/env bash
# Trains shared and per-condition runs of scripts/launch/train_loupe2d_varnet_r8_local_v1_flair_knee.sh at once, one per
# GPU, reruns a run from scratch once if it stops before its last epoch, then scores each seed's shared run against its
# per-condition run with eval_anatomy_pair_ismrm_r8.py (retried on failure, e.g. OOM) into $REPO/$REPORTS. A run whose
# log already reaches the last epoch is not retrained.
#   RUNS   arm:seed:gpu triples (default: shared and anatomy x seeds 42 and 43 on GPUs 0-3)
# Run it from a code snapshot so later edits to the repo cannot reach the runs, reruns or evals:
#   REPO=<repo> nohup setsid <snapshot>/scripts/launch/queue_loupe2d_varnet_r8_local_v1_flair_knee.sh > <log> 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."
CODE="$(pwd)"
PY="${PY:-/data/yanglab2/labmembers2/bkadota/conda/envs/dl/bin/python}"
export PATH="$(dirname "$PY"):$PATH"
VD_POWER="${VD_POWER:-4}"
EPOCHS="${EPOCHS:-100}"
OUT_ROOT="${OUT_ROOT:-$HOME/data/seqmri_checkpoints/local_v1_flair_knee/runs}"
REPO="${REPO:?set REPO to the repository whose reports folder gets the eval JSONs}"
REPORTS="$REPO/${REPORTS:-reports/local_v1_flair_knee}"
RUNS="${RUNS:-shared:42:0 anatomy:42:1 shared:43:2 anatomy:43:3}"
LAST_EPOCH="Epoch = \[ *$EPOCHS/ *$EPOCHS\]"
mkdir -p "$REPORTS"

run_dir() { echo "$OUT_ROOT/loupe2d_varnet_r8_flair_knee_${1}_${EPOCHS}ep_vdinit_p${VD_POWER}_seed${2}"; }

train() {
    local arm=$1 seed=$2 gpu=$3 out
    out="$(run_dir "$arm" "$seed")"
    if grep -qE "$LAST_EPOCH" "$out.log" 2>/dev/null; then
        echo "$(date '+%F %T') $arm seed $seed already finished; not retrained"
        return 0
    fi
    for attempt in 1 2; do
        echo "$(date '+%F %T') start $arm seed $seed GPU $gpu attempt $attempt"
        FOREGROUND=1 GPU="$gpu" ARM="$arm" SEED="$seed" VD_POWER="$VD_POWER" EPOCHS="$EPOCHS" OUT_ROOT="$OUT_ROOT" \
            scripts/launch/train_loupe2d_varnet_r8_local_v1_flair_knee.sh
        if grep -qE "$LAST_EPOCH" "$out.log"; then
            echo "$(date '+%F %T') done $arm seed $seed"
            return 0
        fi
        echo "$(date '+%F %T') $arm seed $seed stopped early; log kept as $out.attempt$attempt.log"
        mv "$out.log" "$out.attempt$attempt.log"
        sleep 120
    done
    return 1
}

seeds=()
for run in $RUNS; do
    IFS=: read -r arm seed gpu <<< "$run"
    train "$arm" "$seed" "$gpu" &
    [[ " ${seeds[*]} " == *" $seed "* ]] || seeds+=("$seed")
done
wait

score() {
    local seed=$1 gpu=$2 tag="vdinit_p${VD_POWER}_seed$1"
    for attempt in 1 2 3; do
        (cd "$REPO" && CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$CODE/scripts/eval/eval_anatomy_pair_ismrm_r8.py" \
            --shared "$(run_dir shared "$seed")" --anatcond "$(run_dir anatomy "$seed")" --tag "$tag" --out "$REPORTS") \
            > "$REPORTS/anatomy_pair_${tag}_stdout.txt" 2>&1
        local rc=$?
        echo "$(date '+%F %T') eval $tag attempt $attempt exit $rc"
        [[ $rc == 0 ]] && return 0
        sleep 300
    done
    return 1
}

gpu=0
for seed in "${seeds[@]}"; do
    if grep -qE "$LAST_EPOCH" "$(run_dir shared "$seed").log" && grep -qE "$LAST_EPOCH" "$(run_dir anatomy "$seed").log"; then
        score "$seed" "$gpu" &
        gpu=$(( (gpu + 1) % 4 ))
    else
        echo "$(date '+%F %T') seed $seed has an unfinished run; not scored"
    fi
done
wait
echo "$(date '+%F %T') queue finished"
