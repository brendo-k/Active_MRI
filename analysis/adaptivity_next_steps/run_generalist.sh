#!/usr/bin/env bash
# Generalist reconstructor: LOUPE run's VarNet fine-tuned on a broad variable-density mask family.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-1}
echo "start generalist $(date)" >> analysis/adaptivity_next_steps/launcher.log
python scripts/diagnostics/generalist_recon_r8.py --out /tmp/bkadota_runs/generalist_vd_varnet_r8 > analysis/adaptivity_next_steps/generalist.log 2>&1
echo "done generalist $(date) rc=$?" >> analysis/adaptivity_next_steps/launcher.log
