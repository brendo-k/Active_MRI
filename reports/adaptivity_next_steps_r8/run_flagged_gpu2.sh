#!/usr/bin/env bash
# Flagged phase (recon.adaptive_input + recon.dc_weight_map): LOUPE + VarNet base from scratch with the plain LOUPE
# run's settings, then the adaptive head from it. Starts after GPU 2's plain runs.
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-2}
FLAGS="recon.adaptive_input=true recon.dc_weight_map=true"
until grep -q "done static_joint_R8" $LOG/launcher.log; do sleep 60; done
echo "start loupe2d_flags_R8 $(date)" >> $LOG/launcher.log
python train.py --model loupe --recon varnet --dataset brain_knee_balanced --acceleration 8 --batch-size 16 --epochs 50 \
  --seed 42 --out-dir $RUNS/loupe2d_flags_R8 sampler.slope=5 sampler.mask_grad_floor=0.05 sampler.ste_grad_floor=0.05 \
  $FLAGS output.run_name=loupe2d_R8_slope5_floor0.05_adaptive-input_dc-weight-map > $LOG/loupe2d_flags_R8.log 2>&1
echo "done loupe2d_flags_R8 $(date) rc=$?" >> $LOG/launcher.log
INIT_MASK=$RUNS/loupe2d_flags_R8/last_model.pt sampler_run adapt16_joint_flags_R8 $RUNS/loupe2d_flags_R8/last_model.pt sampler.adapt_grid=16 $FLAGS
