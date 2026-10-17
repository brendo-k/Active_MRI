#!/usr/bin/env bash
# PromptMR+ adaptive-input VarNet (recon.adaptive_input=true) with the baseline conv_2d sampler, R=8 brain+knee.
# Waits for an empty GPU among 0-2 (GPU 3 stays free) and writes the run to /tmp (/home is nearly full),
# linked into checkpoints/ so the comparison scripts find it.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

NAME=conv2d_varnet_r8_brain_knee_floor0.05_ahain
OUT=/tmp/bkadota_checkpoints/$NAME
ALLOWED_GPUS='^[012]$'
EMPTY_MIB=1000
LOG=analysis/promptmr_ahain/launcher.log

empty_gpu() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F', ' -v allowed="$ALLOWED_GPUS" -v empty="$EMPTY_MIB" '$1 ~ allowed && $2 < empty {print $1; exit}'
}

echo "waiting for an empty GPU in 0-2 $(date)" >> "$LOG"
until GPU=$(empty_gpu) && [ -n "$GPU" ]; do sleep 120; done
sleep 120
GPU=$(empty_gpu)
until [ -n "$GPU" ]; do sleep 120; GPU=$(empty_gpu); done

mkdir -p "$(dirname "$OUT")"
ln -sfn "$OUT" "checkpoints/$NAME"
echo "launching on GPU $GPU $(date)" >> "$LOG"
GPU=$GPU OUT=$OUT scripts/launch/train_conv2d_varnet_r8_brain_knee.sh recon.adaptive_input=true recon.checkpoint_cascades=true >> "$LOG" 2>&1
