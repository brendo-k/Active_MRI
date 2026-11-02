#!/usr/bin/env bash
# Trains the anatomy-mask runs of scripts/launch/train_loupe2d_varnet_r8_vdinit_anatomy.sh at once, one per GPU, reruns a
# run from scratch once if it stops before its last epoch, then scores each seed's shared run against its routed run
# with eval_anatomy_pair_ismrm_r8.py (retried on failure, e.g. OOM), run in $REPO so its
# reports/ismrm2027/anatomy_pair_<tag>.json lands there. A seed is scored once both runs have finished, including a
# shared run trained by an earlier wave.
#   RUNS    arm:seed:gpu triples to train (default: shared and anatomy x seeds 42 and 43 on GPUs 0-3)
#   ROUTED  the arm scored against shared (default anatomy); TAG the eval tag prefix (default vdinit_p<VD_POWER>)
# Run it from a code snapshot so later edits to the repo cannot reach the runs, reruns or evals:
#   REPO=<repo> nohup setsid <snapshot>/scripts/launch/queue_loupe2d_varnet_r8_vdinit_anatomy.sh > <log> 2>&1 &
#   REPO=<repo> VD_POWER=0 RUNS="anatomy_samestart:42:0 anatomy_samestart:43:1" ROUTED=anatomy_samestart \
#       TAG=samestart_p0 nohup setsid <snapshot>/scripts/launch/queue_loupe2d_varnet_r8_vdinit_anatomy.sh > <log> 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."
CODE="$(pwd)"
PY="${PY:-/data/yanglab2/labmembers2/bkadota/conda/envs/dl/bin/python}"
export PATH="$(dirname "$PY"):$PATH"
VD_POWER="${VD_POWER:-4}"
OUT_ROOT="${OUT_ROOT:-$HOME/data/seqmri_checkpoints/vdinit_anatomy/runs}"
REPO="${REPO:?set REPO to the repository whose reports/ismrm2027 gets the eval JSONs}"
REPORTS="$REPO/reports/ismrm2027"
RUNS="${RUNS:-shared:42:0 anatomy:42:1 shared:43:2 anatomy:43:3}"
ROUTED="${ROUTED:-anatomy}"
TAG="${TAG:-vdinit_p$VD_POWER}"
LAST_EPOCH='Epoch = \[ *100/ *100\]'

run_dir() { echo "$OUT_ROOT/loupe2d_varnet_r8_brain_knee_${1}_100ep_vdinit_p${VD_POWER}_seed${2}"; }

train() {
    local arm=$1 seed=$2 gpu=$3 out
    out="$(run_dir "$arm" "$seed")"
    for attempt in 1 2; do
        echo "$(date '+%F %T') start $arm seed $seed GPU $gpu attempt $attempt"
        FOREGROUND=1 GPU="$gpu" ARM="$arm" SEED="$seed" VD_POWER="$VD_POWER" OUT_ROOT="$OUT_ROOT" \
            scripts/launch/train_loupe2d_varnet_r8_vdinit_anatomy.sh
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
    local seed=$1 gpu=$2 tag="${TAG}_seed$1"
    for attempt in 1 2 3; do
        (cd "$REPO" && CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$CODE/scripts/eval/eval_anatomy_pair_ismrm_r8.py" \
            --shared "$(run_dir shared "$seed")" --anatcond "$(run_dir "$ROUTED" "$seed")" --tag "$tag") \
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
    if grep -qE "$LAST_EPOCH" "$(run_dir shared "$seed").log" && grep -qE "$LAST_EPOCH" "$(run_dir "$ROUTED" "$seed").log"; then
        score "$seed" "$gpu" &
        gpu=$(( (gpu + 1) % 4 ))
    else
        echo "$(date '+%F %T') seed $seed has an unfinished run; not scored"
    fi
done
wait
echo "$(date '+%F %T') queue finished"
