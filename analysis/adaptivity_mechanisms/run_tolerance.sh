#!/usr/bin/env bash
# Reconstructor tolerance to mask deviations, then the fine-tune control that splits mismatch from information.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-1}
LOG=analysis/adaptivity_mechanisms
echo "start tolerance $(date)" >> $LOG/launcher.log
python scripts/diagnostics/recon_mask_tolerance_r8.py tolerance > $LOG/tolerance.log 2>&1
echo "done tolerance $(date) rc=$?" >> $LOG/launcher.log
for recon in conv_last loupe_last; do
  echo "start finetune $recon $(date)" >> $LOG/launcher.log
  python scripts/diagnostics/recon_mask_tolerance_r8.py finetune --ft-recon $recon --sigma 1.0 > $LOG/finetune_${recon}_sigma1.log 2>&1
  echo "done finetune $recon $(date) rc=$?" >> $LOG/launcher.log
done
