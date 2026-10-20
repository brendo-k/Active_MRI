#!/usr/bin/env bash
# 2D LOUPE sampler (loupe_2d, straight-through) with a CG-SENSE reconstructor at R=ACCEL (default 8) on balanced
# brain T2 + knee. CG-SENSE has no learned weights, so the reconstructor cannot co-adapt to the mask.
# ANATOMY_MASKS=1 gives each anatomy its own mask (sampler.anatomy_conditioning=true); 0 shares one.
# Run both arms with the same settings: GPU=2 ANATOMY_MASKS=0 ...; GPU=3 ANATOMY_MASKS=1 ...; ACCEL=4 for R=4
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

GPU="${GPU:-0}"
ANATOMY_MASKS="${ANATOMY_MASKS:-0}"
ACCEL="${ACCEL:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
if [ "$ANATOMY_MASKS" = 1 ]; then SUFFIX=anatcond; CONDITIONING=true; else SUFFIX=shared; CONDITIONING=false; fi
OUT="${OUT:-checkpoints/loupe2d_cgsense_r${ACCEL}_brain_knee_$SUFFIX}"

source scripts/launch/_launch.sh

mkdir -p "$(dirname "$OUT")"
launch_detached "$OUT.log" env CUDA_VISIBLE_DEVICES="$GPU" python train.py --model loupe \
    recon=cg_sense \
    datasets=brain_knee_balanced \
    "sampler.accelerations=[$ACCEL.0]" \
    sampler.anatomy_conditioning="$CONDITIONING" \
    training.batch_size="$BATCH_SIZE" \
    output.out_dir="$OUT" "$@"
