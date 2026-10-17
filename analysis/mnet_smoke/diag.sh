#!/usr/bin/env bash
# Short MNet runs at several label settings, to see whether the refined mask ever departs from the mnet mask.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$PYTHONPATH:$(pwd)" WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2
PRE=checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05/best_model.pt
run() { name=$1; shift
  python train.py --model mnet --recon varnet --dataset brain_knee_balanced --acceleration 8 --epochs 1 \
    --out-dir analysis/mnet_smoke/$name --pretrained-recon $PRE training.limit_batches=0.01 "$@" > analysis/mnet_smoke/$name.log 2>&1
  echo "== $name $*"; grep -a "step [0-9]*:" analysis/mnet_smoke/$name.log | cut -c60-330; }
run thr_lr_default
run thr_lr0.05 sampler.mnet_label_lr=0.05
run thr_lr0.05_noreg sampler.mnet_label_lr=0.05 sampler.mnet_sparsity_weight=0 sampler.mnet_consistency_weight=0
run topk_lr0.05_noreg sampler.mnet_label_lr=0.05 sampler.mnet_sparsity_weight=0 sampler.mnet_consistency_weight=0 sampler.mnet_label_binarize=topk
