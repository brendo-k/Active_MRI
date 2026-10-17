#!/usr/bin/env bash
# Poll for an idle GPU, then launch the conv_2d run with a wider score clamp (limit 10, leak 0.05) on it.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
IDLE_MIB=5000
LOG=analysis/clamp_launch/launcher.log

while true; do
    while read -r index used; do
        if [ "$used" -lt "$IDLE_MIB" ]; then
            echo "$(date -Is) launching on GPU $index (used ${used} MiB)" >> "$LOG"
            GPU="$index" OUT=checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_clamp10_leak0.05 \
                bash scripts/launch/train_conv2d_varnet_r8_brain_knee.sh \
                sampler.image_input=false sampler.logit_limit=10 sampler.logit_leak=0.05 >> "$LOG" 2>&1
            echo "$(date -Is) launcher exit $?" >> "$LOG"
            exit 0
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
    sleep 30
done
