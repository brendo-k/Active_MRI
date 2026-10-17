#!/usr/bin/env bash
# Gradient anatomy (stage 1) and transport (stage 2) for the conv sampler and LOUPE, plus reconstructor swaps
# at the conv sampler's operating point. Sequential on one GPU.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-3}
LOG=analysis/adaptivity_mechanisms
CONV=checkpoints/conv2d_varnet_r8_brain_knee_floor0.05
LOUPE=checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05
RANDOM_RECON=checkpoints/random_mask_varnet_r8_brain_knee
run() { tag=$1; shift; echo "start $tag $(date)" >> $LOG/launcher.log
        python scripts/diagnostics/mask_gradient_anatomy_r8.py --tag $tag "$@" > $LOG/gradients_$tag.log 2>&1
        echo "done $tag $(date) rc=$?" >> $LOG/launcher.log; }
run conv_last
run conv_ep4 --sampler-ckpt resume_from_epoch4.pt
run loupe_last --sampler-run $LOUPE
run conv_last_recon_random --recon-run $RANDOM_RECON --recon-ckpt last_model.pt
run conv_last_recon_loupe --recon-run $LOUPE --recon-ckpt last_model.pt
run conv_last_recon_ep4 --recon-run $CONV --recon-ckpt resume_from_epoch4.pt
