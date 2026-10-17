#!/usr/bin/env bash
# Poll for an idle GPU, then launch the conv_2d K=2 stratified-draw run on it.
# GPU 1 is skipped while the queued adaptivity rerun (ADAPTIVITY_PID) is still waiting to use it.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
ADAPTIVITY_PID=1289509
IDLE_MIB=5000
LOG=analysis/mask_draws_launch/launcher.log

while true; do
    while read -r index used; do
        [ "$index" = 1 ] && kill -0 "$ADAPTIVITY_PID" 2>/dev/null && continue
        if [ "$used" -lt "$IDLE_MIB" ]; then
            echo "$(date -Is) launching on GPU $index (used ${used} MiB)" >> "$LOG"
            GPU="$index" OUT=checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_draws2 BATCH_SIZE=8 \
                bash scripts/launch/train_conv2d_varnet_r8_brain_knee.sh sampler.mask_draws=2 sampler.image_input=false >> "$LOG" 2>&1
            echo "$(date -Is) launcher exit $?" >> "$LOG"
            exit 0
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
    sleep 30
done
