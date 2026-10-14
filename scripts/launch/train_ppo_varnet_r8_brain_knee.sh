#!/usr/bin/env bash
# Stage 1 of the PPO arm of the R=8 brain + knee sampler comparison (counterpart of
# train_loupe2d_varnet_r8_brain_knee.sh and train_reinforce2d_varnet_r8_brain_knee.sh).
# PPO learns the adaptive policy against an alternately trained reconstructor; its own
# reconstructor is not comparable to the baselines'. Run train_recon_on_ppo_masks.sh next
# to train a fresh VarNet on the frozen policy's masks with the baselines' schedule.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

GPU="${GPU:-3}"
OUT="${OUT:-checkpoints/ppo_varnet_r8_brain_knee}"

source scripts/launch/_launch.sh

mkdir -p "$(dirname "$OUT")"
launch_detached "$OUT.log" env CUDA_VISIBLE_DEVICES="$GPU" python train.py --model ppo \
    output.out_dir="$OUT" datasets.num_workers=8 "$@"
