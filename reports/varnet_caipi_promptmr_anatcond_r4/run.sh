#!/usr/bin/env bash
# Regenerates eval.json and eval_stdout.txt: scores the four trained checkpoints (cond / blind x seeds 42, 43) at last_model.pt.
# Training the four runs takes hours each and is not rerun here; the commands are below for reference.
#   ARM=cond  GPU=0 SEED=42 scripts/launch/train_varnet_caipi_promptmr_anatcond.sh
#   ARM=blind GPU=1 SEED=42 scripts/launch/train_varnet_caipi_promptmr_anatcond.sh
#   ARM=cond  GPU=2 SEED=43 scripts/launch/train_varnet_caipi_promptmr_anatcond.sh
#   ARM=blind GPU=3 SEED=43 scripts/launch/train_varnet_caipi_promptmr_anatcond.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CUDA_VISIBLE_DEVICES=2 python scripts/eval/eval_varnet_caipi_anatcond.py --seeds 42 43 2>&1 | tee reports/varnet_caipi_promptmr_anatcond_r4/eval_stdout.txt
