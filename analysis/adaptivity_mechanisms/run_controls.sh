#!/usr/bin/env bash
# Transport with the population / anatomy push controls (from saved gradients), and the reverse pull: LOUPE masks
# through the conv reconstructor.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-1}
LOG=analysis/adaptivity_mechanisms
CONV=checkpoints/conv2d_varnet_r8_brain_knee_floor0.05
LOUPE=checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05
grads() { tag=$1; shift; echo "start controls $tag $(date)" >> $LOG/launcher.log
          python scripts/diagnostics/mask_gradient_anatomy_r8.py --tag $tag "$@" > $LOG/gradients_$tag.log 2>&1
          echo "done controls $tag $(date) rc=$?" >> $LOG/launcher.log; }
grads loupe_last_recon_conv --sampler-run $LOUPE --recon-run $CONV --recon-ckpt last_model.pt
grads conv_last --from-saved
grads conv_ep4 --sampler-ckpt resume_from_epoch4.pt --from-saved
