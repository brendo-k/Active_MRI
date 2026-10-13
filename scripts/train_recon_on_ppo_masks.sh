#!/usr/bin/env bash
# Stage 2 of the PPO arm of the R=8 brain + knee sampler comparison: a fresh VarNet and
# sensitivity net trained from scratch on the masks of a frozen, greedy PPO policy, with
# the same data, batch size, epochs and learning-rate schedule as
# train_loupe2d_varnet_r8_brain_knee.sh (50 epochs; add training.num_epochs=100
# training.lr_step_size=90 for the 100-epoch REINFORCE schedule).
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

GPU="${GPU:-3}"
PPO_POLICY="${PPO_POLICY:-checkpoints/ppo_varnet_r8_brain_knee/best.pt}"
OUT="${OUT:-checkpoints/ppo_policy_varnet_r8_brain_knee}"
BATCH_SIZE="${BATCH_SIZE:-16}"

source scripts/_launch.sh

mkdir -p "$(dirname "$OUT")"
launch_detached "$OUT.log" env CUDA_VISIBLE_DEVICES="$GPU" python scripts/train_sequential.py \
    recon=varnet \
    datasets=brain_knee_balanced \
    'sampler.accelerations=[8.0]' \
    sampler.sampler_type=loupe_2d \
    sampler.ppo_policy="$PPO_POLICY" \
    training.batch_size="$BATCH_SIZE" \
    output.out_dir="$OUT" "$@"
