#!/usr/bin/env bash
# Trains the four spacing arms of scripts/launch/train_varnet_r8_spacing_masks.sh at once, one per GPU, reruns a run
# from scratch once if it stops before its last epoch, then trains top-k on GPU 0 while scoring the four on GPU 1, and
# scores all five once top-k is done, with scripts/eval/eval_spacing_masks_r8.py (retried on failure, e.g. OOM), run in
# $REPO so its reports/mask_spacing_r8/eval_spacing{,_topk}.json land there.
# Run it from a code snapshot so later edits to the repo cannot reach the runs, reruns or eval:
#   REPO=<repo> nohup setsid <snapshot>/scripts/launch/queue_varnet_r8_spacing_masks.sh > <log> 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."
CODE="$(pwd)"
PY="${PY:-/data/yanglab2/labmembers2/bkadota/conda/envs/dl/bin/python}"
export PATH="$(dirname "$PY"):$PATH"
OUT_ROOT="${OUT_ROOT:-$HOME/data/seqmri_checkpoints/spacing_r8/runs}"
REPO="${REPO:?set REPO to the repository whose reports/mask_spacing_r8 gets the eval JSON}"
LAST_EPOCH='Epoch = \[ *50/ *50\]'

train() {
    local arm=$1 gpu=$2 out="$OUT_ROOT/varnet_r8_brain_knee_fixedmask_${1}_50ep_seed42"
    for attempt in 1 2; do
        echo "$(date '+%F %T') start $arm GPU $gpu attempt $attempt"
        FOREGROUND=1 GPU="$gpu" ARM="$arm" OUT_ROOT="$OUT_ROOT" scripts/launch/train_varnet_r8_spacing_masks.sh
        if grep -qE "$LAST_EPOCH" "$out.log"; then
            echo "$(date '+%F %T') done $arm"
            return 0
        fi
        echo "$(date '+%F %T') $arm stopped early; log kept as $out.attempt$attempt.log"
        mv "$out.log" "$out.attempt$attempt.log"
        sleep 120
    done
    return 1
}

score() {
    local tag=$1 gpu=$2
    for attempt in 1 2 3; do
        (cd "$REPO" && CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$CODE/scripts/eval/eval_spacing_masks_r8.py" \
            --runs-root "$OUT_ROOT" --tag "$tag") > "$REPO/reports/mask_spacing_r8/eval_${tag}_stdout.txt" 2>&1
        local rc=$?
        echo "$(date '+%F %T') eval $tag attempt $attempt exit $rc"
        [[ $rc == 0 ]] && return 0
        sleep 300
    done
    return 1
}

train pd_loupe_s0 0 &
train pd_loupe_s1 1 &
train white_loupe_s0 2 &
train white_loupe_s1 3 &
wait
train topk_loupe 0 &
score spacing 1
wait
score spacing_topk 1
echo "$(date '+%F %T') queue finished"
