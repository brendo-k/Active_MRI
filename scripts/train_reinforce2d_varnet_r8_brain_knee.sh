#!/usr/bin/env bash
# 2D REINFORCE sampler (loupe_2d + mask_train_mode=reinforce) with a VarNet
# reconstructor at R=8 on balanced brain T2 + knee (fat-sat and non-fat-sat).
#
# Set PRETRAINED_RECON to an R=8 VarNet checkpoint to warm-start and freeze the
# reconstructor; otherwise the reconstructor trains jointly from scratch.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

GPU="${GPU:-1}"
OUT="${OUT:-checkpoints/reinforce2d_varnet_r8_brain_knee}"
PRETRAINED_RECON="${PRETRAINED_RECON:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-0.005}"
NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-true}"

FREEZE_ARGS=()
if [[ -n "$PRETRAINED_RECON" ]]; then
    FREEZE_ARGS=(output.pretrained_recon="$PRETRAINED_RECON" sampler.freeze_recon=true)
fi

source scripts/_launch.sh

mkdir -p "$(dirname "$OUT")"
launch_detached "$OUT.log" env CUDA_VISIBLE_DEVICES="$GPU" python scripts/train_sequential.py \
    recon=varnet \
    datasets=brain_knee_balanced \
    'sampler.accelerations=[8.0]' \
    sampler.sampler_type=loupe_2d \
    sampler.mask_train_mode=reinforce \
    sampler.reinforce_entropy_weight="$ENTROPY_WEIGHT" \
    sampler.reinforce_normalize_advantages="$NORMALIZE_ADVANTAGES" \
    training.batch_size="$BATCH_SIZE" \
    output.out_dir="$OUT" "${FREEZE_ARGS[@]}" "$@"
