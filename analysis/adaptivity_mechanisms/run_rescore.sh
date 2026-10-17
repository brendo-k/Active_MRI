#!/usr/bin/env bash
# Re-score saved stage-1 gradients with the final analysis (scales, smooth cosines); transport for the conv runs.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-1}
LOG=analysis/adaptivity_mechanisms
RANDOM_RECON=checkpoints/random_mask_varnet_r8_brain_knee
grads() { tag=$1; shift; echo "start rescore $tag $(date)" >> $LOG/launcher.log
          python scripts/diagnostics/mask_gradient_anatomy_r8.py --tag $tag --from-saved "$@" > $LOG/gradients_$tag.log 2>&1
          echo "done rescore $tag $(date) rc=$?" >> $LOG/launcher.log; }
grads conv_last
grads conv_ep4 --sampler-ckpt resume_from_epoch4.pt
grads conv_last_recon_random --recon-run $RANDOM_RECON --recon-ckpt last_model.pt
