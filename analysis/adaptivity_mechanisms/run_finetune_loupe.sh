#!/usr/bin/env bash
# Fine-tune control for the LOUPE reconstructor (mismatch vs information of a smooth reshape, sigma 1).
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-2}
LOG=analysis/adaptivity_mechanisms
echo "start finetune loupe_last $(date)" >> $LOG/launcher.log
python scripts/diagnostics/recon_mask_tolerance_r8.py finetune --ft-recon loupe_last --sigma 1.0 > $LOG/finetune_loupe_last_sigma1.log 2>&1
echo "done finetune loupe_last $(date) rc=$?" >> $LOG/launcher.log
