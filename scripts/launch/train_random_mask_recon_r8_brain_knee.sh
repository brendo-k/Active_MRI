#!/usr/bin/env bash
# VarNet trained on uniformly random R=8 masks (brain T2 + knee, balanced): the warm-start
# reconstructor U0 of the MNet alternating run. The MNet quality check scores a random mask with
# this frozen network, and mask-backward starts from its weights.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

GPU="${GPU:-0}"
OUT="${OUT:-checkpoints/random_mask_varnet_r8_brain_knee}"
BATCH_SIZE="${BATCH_SIZE:-8}"

source scripts/launch/_launch.sh

mkdir -p "$(dirname "$OUT")"
launch_detached "$OUT.log" env CUDA_VISIBLE_DEVICES="$GPU" python train.py --model loupe \
    recon=varnet \
    datasets=brain_knee_balanced \
    training.num_step=1 \
    'sampler.accelerations=[8.0]' \
    sampler.mask_train_mode=random \
    training.batch_size="$BATCH_SIZE" \
    training.seed=42 \
    output.out_dir="$OUT" "$@"
