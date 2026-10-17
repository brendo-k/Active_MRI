#!/usr/bin/env bash
# MNet (the conv_2d U-Net with exact-budget top-k) trained by the alternating mask-backward /
# sampler-fit scheme of Huang & Ravishankar (TCI 2022), VarNet reconstructor, R=8, balanced brain T2 + knee.
# Data, acceleration, reconstructor, epochs and seed match train_conv2d_varnet_r8_brain_knee.sh;
# batch size is 8 (rerun the conv baseline with BATCH_SIZE=8 for a matched comparison). Project-default
# Adam lr throughout. PRETRAINED_RECON must be a reconstructor trained on random masks
# (train_random_mask_recon_r8_brain_knee.sh). Axis flags, appended as Hydra overrides:
#   sampler.mnet_accept=always           train MNet on every refined mask (no quality check)
#   sampler.mnet_compare_random=false    drop the random-mask arm of the check
#   sampler.mnet_rollback=false          keep the reconstructor update of rejected batches
#   sampler.mnet_degeneracy_fallback=false
#   sampler.mnet_label_steps=T sampler.mnet_sampler_steps=c sampler.mnet_label_lr=...
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

GPU="${GPU:-0}"
OUT="${OUT:-checkpoints/mnet_varnet_r8_brain_knee}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PRETRAINED_RECON="${PRETRAINED_RECON:?set PRETRAINED_RECON to a random-mask-trained reconstructor checkpoint}"

source scripts/launch/_launch.sh

mkdir -p "$(dirname "$OUT")"
launch_detached "$OUT.log" env CUDA_VISIBLE_DEVICES="$GPU" python train.py --model mnet \
    recon=varnet \
    datasets=brain_knee_balanced \
    training.num_step=1 \
    'sampler.accelerations=[8.0]' \
    training.batch_size="$BATCH_SIZE" \
    training.seed=42 \
    output.pretrained_recon="$PRETRAINED_RECON" \
    output.out_dir="$OUT" "$@"
