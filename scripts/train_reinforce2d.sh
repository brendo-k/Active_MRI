#!/usr/bin/env bash
# Train the 2D REINFORCE sampler (loupe_2d + mask_train_mode=reinforce) on
# combined brain+knee fastMRI data. See docs/GREEDY_VS_LEARNED_SAMPLING.md
# for what this training mode does and why.
#
# Set PRETRAINED_RECON to a stage-1 checkpoint (e.g. produced by a
# straight_through run) to warm-start and freeze the reconstructor, so
# REINFORCE only has to learn the sampling policy instead of splitting
# gradient signal with a reconstructor that is also still training.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

RECON="${RECON:-varnet}"
GPU="${GPU:-0}"
OUT="${OUT:-checkpoints/reinforce2d_${RECON}_combined14_1s}"
PRETRAINED_RECON="${PRETRAINED_RECON:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.005}"
NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-true}"

FREEZE_ARGS=()
if [[ -n "$PRETRAINED_RECON" ]]; then
    FREEZE_ARGS=(output.pretrained_recon="$PRETRAINED_RECON" sampler.freeze_recon=true)
fi

CUDA_VISIBLE_DEVICES=$GPU python scripts/train_sequential.py \
    recon=$RECON \
    datasets=combined14 \
    sampler.sampler_type=loupe_2d \
    sampler.mask_train_mode=reinforce \
    sampler.reinforce_entropy_weight=$ENTROPY_WEIGHT \
    sampler.reinforce_normalize_advantages=$NORMALIZE_ADVANTAGES \
    training.batch_size=$BATCH_SIZE \
    output.out_dir=$OUT "${FREEZE_ARGS[@]}" "$@" \
    > "$OUT.log" 2>&1 &
PID=$!

echo "PID: $PID"
echo "Log: $OUT.log"
wait $PID
