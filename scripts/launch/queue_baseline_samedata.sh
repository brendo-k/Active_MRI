#!/usr/bin/env bash
# Waits for the FiLM run to exit, then trains the unconditioned CAIPI VarNet on the same
# (PD/PDFS-aware) knee loader at the baseline lr, as the control for the FiLM comparison.
set -euo pipefail

cd "$(dirname "$0")/../.."
while pgrep -f "output.out_dir=checkpoints/varnet_caipi2d_r4_brain_knee_film$" > /dev/null; do sleep 30; done

scripts/launch/train_varnet_caipi_brain_knee.sh \
    training.lr=0.001 \
    output.out_dir=checkpoints/varnet_caipi2d_r4_brain_knee_samedata
