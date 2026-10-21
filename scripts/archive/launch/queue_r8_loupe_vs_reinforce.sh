#!/usr/bin/env bash
# Queues the two R=8 brain+knee VarNet runs for a like-for-like comparison:
#   1. straight-through LOUPE        -> checkpoints/loupe2d_varnet_r8_brain_knee
#   2. 2D REINFORCE (policy lr 0.02, 4 slices x 4 draws per batch)
#                                    -> checkpoints/reinforce2d_varnet_r8_brain_knee_v2
# Each job starts on the first GPU whose memory use is low enough to fit it beside
# whatever is already running there. Run with: nohup scripts/launch/queue_r8_loupe_vs_reinforce.sh &
set -euo pipefail

cd "$(dirname "$0")/../.."

MAX_USED_MIB="${MAX_USED_MIB:-19000}"  # a batch-16 VarNet run needs ~30 GB of the 48 GB card
POLL_SECONDS=60
SETTLE_SECONDS=300  # let a new job allocate before the next GPU check

first_free_gpu() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F', ' -v max="$MAX_USED_MIB" '$2 <= max && !found {print $1; found = 1}'
}

launch_when_free() {
    local name="$1"; shift
    local gpu=""
    until gpu="$(first_free_gpu)" && [[ -n "$gpu" ]]; do sleep "$POLL_SECONDS"; done
    echo "$(date '+%F %T') launching $name on GPU $gpu"
    GPU="$gpu" "$@"
    sleep "$SETTLE_SECONDS"
}

launch_when_free loupe scripts/launch/train_loupe2d_varnet_r8_brain_knee.sh datasets.num_workers=6

launch_when_free reinforce env OUT=checkpoints/reinforce2d_varnet_r8_brain_knee_v2 \
    scripts/launch/train_reinforce2d_varnet_r8_brain_knee.sh \
    sampler.reinforce_lr=0.02 sampler.reinforce_group_size=4 datasets.num_workers=6

echo "$(date '+%F %T') both jobs launched"
