#!/usr/bin/env bash
# Does 1-px sample spacing matter for VarNet at R=8 (brain T2 + knee)? The local run of the HPC spacing arms
# (scripts/launch/hpc/spacing_vdinit_r8/run.md, tasks 4-8; the HPC array never ran).
#
# The learned LOUPE mask's pixel spacing matches an independent draw (1-px pair correlation 0.86 vs 0.87). Every arm
# trains a VarNet (and its sensitivity net) from scratch on one fixed mask made by
# scripts/diagnostics/make_spacing_masks_r8.py from the learned density of
# checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05: same 10x10 centre, same ring-by-ring density, 8,192
# samples (Sparsity 0.125). Training seed 42 in every arm, so the arms share the reconstructor init and batch order and
# differ only in the mask.
#   ARM=pd_loupe_s0 / pd_loupe_s1        Poisson-disc realisation of the density, mask seed 0 / 1
#   ARM=white_loupe_s0 / white_loupe_s1  independent draw from the same density, mask seed 0 / 1
#   ARM=topk_loupe                       the highest-probability locations, deterministic (moves samples inward)
# Recipe of the reference run and the HPC array: 12-cascade VarNet, bs 16, 50 epochs, lr 1e-3 stepped at epoch 40,
# on the repo's current defaults (bf16 cascades and sens net, compiled U-Nets, cuDNN autotuner, padded-coil skip).
#
# Eval: scripts/eval/eval_spacing_masks_r8.py (each reconstructor on its own mask and on the others, paired per val
# slice, 95% volume bootstrap, PSNR dB).
# Decision rule, fixed before any result (pd - white per mask seed, own reconstructor + own mask, per anatomy):
#   no spacing effect   both anatomies within +-0.05 dB, or the two mask seeds disagree in sign
#   spacing matters     Poisson-disc ahead by >= 0.1 dB in both mask seeds and both anatomies (then a position-aware
#                       learner, BASS- or AutoSamp-style, is worth building)
#   mixed               anything else: report per seed, claim nothing
# Top-k: expected to trail white_loupe_s0; that would show determinism alone does not help.
#
#   ARM=pd_loupe_s0 GPU=0 scripts/launch/train_varnet_r8_spacing_masks.sh
# FOREGROUND=1 runs the job in this shell (for scripts/launch/queue_varnet_r8_spacing_masks.sh).
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

ARM="${ARM:?set ARM=pd_loupe_s0, pd_loupe_s1, white_loupe_s0, white_loupe_s1 or topk_loupe}"
GPU="${GPU:-0}"
MASK="$(pwd)/reports/mask_spacing_r8/masks_r8/$ARM.pt"
OUT_ROOT="${OUT_ROOT:-$HOME/data/seqmri_checkpoints/spacing_r8/runs}"
[[ -f "$MASK" ]] || { echo "no mask $MASK" >&2; exit 1; }

OUT="$OUT_ROOT/varnet_r8_brain_knee_fixedmask_${ARM}_50ep_seed42"
mkdir -p "$OUT_ROOT"

CMD=(env CUDA_VISIBLE_DEVICES="$GPU" python train.py --model sequential \
    sampler.sampler_type=loupe_2d \
    sampler.fixed_mask_path="$MASK" \
    recon=varnet \
    datasets=brain_knee_balanced \
    'sampler.accelerations=[8.0]' \
    training.batch_size=16 \
    training.num_epochs=50 \
    training.lr_step_size=40 \
    training.seed=42 \
    output.out_dir="$OUT" "$@")

if [[ "${FOREGROUND:-0}" == 1 ]]; then
    "${CMD[@]}" > "$OUT.log" 2>&1
else
    source scripts/launch/_launch.sh
    launch_detached "$OUT.log" "${CMD[@]}"
fi
