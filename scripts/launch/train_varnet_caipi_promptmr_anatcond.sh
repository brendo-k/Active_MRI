#!/usr/bin/env bash
# Does conditioning on the anatomy help or hurt a PromptMR+-style VarNet on a fixed CAIPI pattern?
#
# VarNet (12 cascades, chans 18) on 2D-CAIPI 2x2 shift 1 (R=4, plus the 10x10 calibration centre), balanced brain T2 +
# knee, with PromptMR+'s adaptive input (A^H A x, A^H y, carried latent), PromptMR+'s momentum layer (every decoder
# level, the 11 earlier cascades) and a PromptMR-style prompt at every decoder level, over the classes
# [brain, knee-PD, knee-PDFS].
#
# Two arms with identical code, parameters, initialisation and data order; only the label differs:
#   cond   each slice's class selects its prompt, on top of the features
#   blind  every slice is fed class 0, so the prompt sees the features alone (PromptMR's implicit prompt)
# lr 1e-3 stepped at epoch 40, 50 epochs, bs 4, as the earlier CAIPI runs (plain rbdddrmf, bottleneck FiLM qpaflt54).
# Score with scripts/eval/eval_varnet_caipi_anatcond.py (paired per class, plus label swaps on the cond arm).
#
#   ARM=cond GPU=2 SEED=42 scripts/launch/train_varnet_caipi_promptmr_anatcond.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

ARM="${ARM:?set ARM=cond or ARM=blind}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
case "$ARM" in
    cond) BLIND=false ;;
    blind) BLIND=true ;;
    *) echo "ARM must be cond or blind, got $ARM" >&2; exit 1 ;;
esac

NAME="varnet_caipi2d_r4_promptmr_${ARM}_seed${SEED}"
OUT="/tmp/bkadota_checkpoints/varnet_caipi_promptmr_anatcond/$NAME"  # /home is nearly full
mkdir -p "$(dirname "$OUT")"
ln -sfn "$OUT" "checkpoints/$NAME"

source scripts/launch/_launch.sh

launch_detached "$OUT.log" env CUDA_VISIBLE_DEVICES="$GPU" python train.py --model sequential \
    recon=varnet \
    recon.adaptive_input=true \
    recon.momentum_history=11 \
    recon.prompt_conditioning=true \
    recon.label_blind="$BLIND" \
    datasets=brain_knee_balanced \
    'datasets.condition_classes=[brain,knee-PD,knee-PDFS]' \
    training.num_step=1 \
    sampler.sampler_type=caipi_2d \
    training.lr=0.001 \
    training.lr_scheduler=step \
    training.lr_step_size=40 \
    training.num_epochs=50 \
    training.batch_size=4 \
    training.seed="$SEED" \
    output.out_dir="$OUT" "$@"
