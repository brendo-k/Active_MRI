#!/usr/bin/env bash
# One run of the PPO observation comparison (kspace | aligned | image | combined) at lr 3e-3, policy-only
# (no recon phase), best.pt chosen on sampled validation draws. Usage:
#   GPU=1 scripts/train_ppo_obs_arm.sh <arm> <seed> [hydra overrides...]
# Env: OUT_ROOT (default checkpoints/ppo_obs_v2), STEPS (default 12600 env steps = ~100 updates).
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

ARM="$1"; SEED="$2"; shift 2
GPU="${GPU:-0}"
OUT="${OUT_ROOT:-checkpoints/ppo_obs_v2}/${ARM}_s${SEED}"
STEPS="${STEPS:-12600}"

case "$ARM" in
    kspace)   OBS=kspace;               NORM=false ;;
    aligned)  OBS=kspace_aligned;       NORM=true ;;
    image)    OBS=image;                NORM=true ;;
    combined) OBS=image_kspace_aligned; NORM=true ;;
    *) echo "unknown arm: $ARM (kspace|aligned|image|combined)" >&2; exit 1 ;;
esac

source scripts/_launch.sh

mkdir -p "$(dirname "$OUT")"
launch_detached "$OUT.log" env CUDA_VISIBLE_DEVICES="$GPU" python scripts/train_ppo.py \
    output.out_dir="$OUT" training.seed="$SEED" env.observation="$OBS" ppo.normalize_obs="$NORM" \
    datasets.num_workers=4 ppo.num_rounds=1 ppo.steps_per_round="$STEPS" ppo.recon_steps_per_round=0 \
    ppo.learning_rate=3e-3 "$@"
