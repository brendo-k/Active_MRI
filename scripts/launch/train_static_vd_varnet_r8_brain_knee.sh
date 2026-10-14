#!/usr/bin/env bash
# Static variable-density 2D mask at R=8 with a VarNet on balanced brain T2 + knee, for the
# learning-rate / scheduler sweep. The mask is AutoSamp's truncated-normal draw with the
# positions frozen (autosamp_lr=0, no anneal), so it is one fixed pattern that the seed makes
# identical across runs; only the reconstructor trains.
#
#   LR=1e-3 SCHED=cosine EPOCHS=50 GPU=3 scripts/launch/train_static_vd_varnet_r8_brain_knee.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

GPU="${GPU:-3}"
LR="${LR:-0.0003}"
SCHED="${SCHED:-step}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-42}"
OUT="${OUT:-checkpoints/lr_sweep/static_vd_lr${LR}_${SCHED}_${EPOCHS}ep_seed${SEED}}"

source scripts/launch/_launch.sh

mkdir -p "$(dirname "$OUT")"
launch_detached "$OUT.log" env CUDA_VISIBLE_DEVICES="$GPU" python train.py --model sequential \
    recon=varnet \
    datasets=brain_knee_balanced \
    'sampler.accelerations=[8.0]' \
    sampler.sampler_type=autosamp_2d \
    sampler.autosamp_init=normal \
    sampler.autosamp_lr=0 \
    sampler.autosamp_anneal_fraction=0 \
    training.batch_size="$BATCH_SIZE" \
    training.lr="$LR" \
    training.lr_scheduler="$SCHED" \
    training.num_epochs="$EPOCHS" \
    training.seed="$SEED" \
    output.out_dir="$OUT" "$@"
