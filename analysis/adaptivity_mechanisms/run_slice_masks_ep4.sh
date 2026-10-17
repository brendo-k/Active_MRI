#!/usr/bin/env bash
# Per-slice vs shared free masks through the conv run's epoch-4 reconstructor (run beside the GPU 3 queue).
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-1}
LOG=analysis/adaptivity_mechanisms
echo "start slice_masks conv_ep4 $(date)" >> $LOG/launcher.log
python scripts/diagnostics/recon_bias_slice_masks_r8.py --tag conv_ep4 --chunk 4 --recon-run checkpoints/conv2d_varnet_r8_brain_knee_floor0.05 --recon-ckpt resume_from_epoch4.pt > $LOG/slice_masks_conv_ep4.log 2>&1
echo "done slice_masks conv_ep4 $(date) rc=$?" >> $LOG/launcher.log
