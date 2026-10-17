#!/usr/bin/env bash
# Privileged control on GPU 0: the predictor sees the fully sampled k-space. Mirrors the observed-input screens
# gain_lr1e-3_dev{0,1}, so the gap between the two says whether the labels are unlearnable or merely unobservable.
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=0
DIR=analysis/label_gonogo/reweighted
mkdir -p $DIR
for dev in 0 1; do
    name=full_gain_lr1e-3_dev$dev
    python scripts/diagnostics/label_predictor_gonogo_r8.py --screen --predictor-input full --weight gain --lr 1e-3 \
        --deviation-weight $dev --out $DIR/privileged_$name.json > $DIR/privileged_$name.log 2>&1
    echo "privileged $name done $(date)" >> analysis/label_gonogo/launcher.log
done
