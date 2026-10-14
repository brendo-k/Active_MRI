#!/usr/bin/env bash
# Train one-step multi-anatomy (brain+knee) models with anatomy classification
# on U-Net, VarNet, and CG-SENSE reconstructors in parallel on separate GPUs.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/../.." && pwd)"

# ANATOMY_CONDITIONING=true feeds the anatomy label to the sampler (and, for
# VarNet, to every cascade) so the learned pattern can differ per anatomy.
COND="${ANATOMY_CONDITIONING:-false}"

COMMON="datasets=multi_anatomy training.num_step=0 loss.anatomy_weight=0.1"
COMMON="$COMMON datasets.data_dir=/home/bkadota/data/fastmri/combined_14"
COMMON="$COMMON sampler.anatomy_conditioning=$COND"

echo "=== Launching U-Net on GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 python train.py --model conv recon=unet $COMMON \
    output.out_dir=checkpoints/multitask_unet "$@" \
    > checkpoints/multitask_unet.log 2>&1 &
PID_UNET=$!

echo "=== Launching VarNet on GPU 1 ==="
CUDA_VISIBLE_DEVICES=1 python train.py --model conv recon=varnet $COMMON \
    recon.anatomy_conditioning=$COND \
    output.out_dir=checkpoints/multitask_varnet "$@" \
    > checkpoints/multitask_varnet.log 2>&1 &
PID_VARNET=$!

echo "=== Launching CG-SENSE on GPU 2 ==="
CUDA_VISIBLE_DEVICES=2 python train.py --model conv recon=cg_sense $COMMON \
    output.out_dir=checkpoints/multitask_cgsense "$@" \
    > checkpoints/multitask_cgsense.log 2>&1 &
PID_CGSENSE=$!

echo "PIDs: U-Net=$PID_UNET VarNet=$PID_VARNET CG-SENSE=$PID_CGSENSE"
echo "Logs: checkpoints/multitask_{unet,varnet,cgsense}.log"
echo "Waiting for all jobs..."
wait $PID_UNET $PID_VARNET $PID_CGSENSE
echo "All training jobs complete."
