#!/usr/bin/env bash
# Per-slice vs shared free masks through four frozen reconstructors; three at a time on one GPU.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-3}
LOG=analysis/adaptivity_mechanisms
CONV=checkpoints/conv2d_varnet_r8_brain_knee_floor0.05
LOUPE=checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05
RANDOM_RECON=checkpoints/random_mask_varnet_r8_brain_knee
masks() { tag=$1; shift; echo "start slice_masks $tag $(date)" >> $LOG/launcher.log
          python scripts/diagnostics/recon_bias_slice_masks_r8.py --tag $tag --chunk 4 "$@" > $LOG/slice_masks_$tag.log 2>&1
          echo "done slice_masks $tag $(date) rc=$?" >> $LOG/launcher.log; }
masks conv_last &
masks random --recon-run $RANDOM_RECON &
(masks loupe_last --recon-run $LOUPE; masks conv_ep4 --recon-run $CONV --recon-ckpt resume_from_epoch4.pt) &
wait
