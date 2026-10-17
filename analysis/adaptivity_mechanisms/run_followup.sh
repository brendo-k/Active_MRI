#!/usr/bin/env bash
# Follow-up on one GPU: random-mask reconstructor gradients, re-analysis of saved gradients with the final metrics,
# then per-slice vs shared free-mask fits through four frozen reconstructors.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-3}
LOG=analysis/adaptivity_mechanisms
CONV=checkpoints/conv2d_varnet_r8_brain_knee_floor0.05
LOUPE=checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05
RANDOM_RECON=checkpoints/random_mask_varnet_r8_brain_knee
grads() { tag=$1; shift; echo "start gradients $tag $(date)" >> $LOG/launcher.log
          python scripts/diagnostics/mask_gradient_anatomy_r8.py --tag $tag "$@" > $LOG/gradients_$tag.log 2>&1
          echo "done gradients $tag $(date) rc=$?" >> $LOG/launcher.log; }
masks() { tag=$1; shift; echo "start slice_masks $tag $(date)" >> $LOG/launcher.log
          python scripts/diagnostics/recon_bias_slice_masks_r8.py --tag $tag "$@" > $LOG/slice_masks_$tag.log 2>&1
          echo "done slice_masks $tag $(date) rc=$?" >> $LOG/launcher.log; }
grads conv_last_recon_random --recon-run $RANDOM_RECON --recon-ckpt last_model.pt
grads conv_last --from-saved
grads conv_ep4 --sampler-ckpt resume_from_epoch4.pt --from-saved
grads loupe_last --sampler-run $LOUPE --from-saved
grads conv_last_recon_loupe --recon-run $LOUPE --recon-ckpt last_model.pt --from-saved
grads conv_last_recon_ep4 --recon-run $CONV --recon-ckpt resume_from_epoch4.pt --from-saved
masks conv_last
masks random --recon-run $RANDOM_RECON
masks loupe_last --recon-run $LOUPE
masks conv_ep4 --recon-run $CONV --recon-ckpt resume_from_epoch4.pt
