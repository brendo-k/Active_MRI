#!/usr/bin/env bash
# Deviation-only predictor on GPU 1: memorisation tests first (can any input fit 64 rows?), then the
# coarse-to-fine curriculum against a no-curriculum control on all training rows.
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=1
DIR=analysis/label_gonogo/curriculum
mkdir -p $DIR
RUNS=(
    "--input acs_image --memorise 64 --sigmas 0 --epochs-per-stage 150"
    "--input kspace_log --memorise 64 --sigmas 0 --epochs-per-stage 150"
    "--input kspace --memorise 64 --sigmas 0 --epochs-per-stage 150"
    "--input acs_image --full --memorise 64 --sigmas 0 --epochs-per-stage 150"
    "--input acs_image --sigmas 8 4 2 1 0 --epochs-per-stage 8"
    "--input acs_image --sigmas 0 --epochs-per-stage 40"
    "--input kspace_log --sigmas 8 4 2 1 0 --epochs-per-stage 8"
    "--input acs_image --full --sigmas 8 4 2 1 0 --epochs-per-stage 8"
)
for options in "${RUNS[@]}"; do
    name=$(echo $options | tr -d '-' | tr ' ' '_')
    python scripts/diagnostics/label_curriculum_r8.py $options > $DIR/$name.log 2>&1
    echo "curriculum $name done $(date)" >> analysis/label_gonogo/launcher.log
done
