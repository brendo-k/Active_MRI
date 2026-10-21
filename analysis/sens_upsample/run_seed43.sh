#!/bin/bash
# Second seed for the conv vs bilinear sensitivity-U-Net upsampling pair (CG-SENSE, LOUPE 2D, R=8, 50 epochs).
# Each job waits for a GPU with enough free memory at its own start time (peak ~21 GB, +margin).
cd /home/bkadota/code/active_mri/SeqMRI
source .venv/bin/activate
NEEDED_MIB=23000

free_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F', ' -v need=$NEEDED_MIB '$2 >= need {print $1; exit}'
}

run_arm() {
  name=$1; shift
  until gpu=$(free_gpu); [ -n "$gpu" ]; do sleep 120; done
  echo "start $name on GPU $gpu at $(date)"
  CUDA_VISIBLE_DEVICES=$gpu python train.py --model loupe --recon cg_sense --dataset brain_knee_balanced \
    --acceleration 8 --batch-size 16 --epochs 50 --seed 43 \
    --out-dir checkpoints/$name "$@" > checkpoints/$name.log 2>&1
  rc=$?
  echo "done $name rc=$rc at $(date)"
}

run_arm loupe2d_cgsense_r8_brain_knee_shared_seed43 sampler.sens_upsample_method=conv &
sleep 240
run_arm loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep_seed43 sampler.sens_upsample_method=bilinear &
wait
