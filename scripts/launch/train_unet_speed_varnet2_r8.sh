#!/usr/bin/env bash
# Can the VarNet's U-Net be made faster without losing reconstruction quality? A 2-cascade VarNet (chans 18, 4 pools)
# on the R=8 static variable-density mask of the 12-cascade static-VD baseline (lr_sweep/static_vd_lr0.001_step_50ep_seed42,
# saved by scripts/diagnostics/export_static_vd_mask_r8.py), balanced brain T2 + knee, lr 1e-3 stepped at epoch 40,
# 50 epochs, bs 16: the static-VD baseline recipe with 2 cascades. The mask is a file rather than the seed's AutoSamp
# draw because that draw comes after the reconstructor's initialisation, so it would differ between U-Nets. Every arm
# and seed sees the same mask; the seed sets the initialisation and data order. Arms differ only in the U-Nets:
#   base       fastMRI's U-Net with its LeakyReLUs out of place (VarNet.FastMRIUnet; same outputs)
#   bf16       base under bf16 autocast (normalisation around it, FFTs and DC stay fp32)
#   sens64     base with the sensitivity net's U-Net run on the 64x64 centre of k-space (its input is only the 10x10
#              centre) and its output upsampled to the full image; the cascades are base's. Half the epoch time, but
#              -0.10 dB brain, -0.21 dB knee over 3 seeds (eval_sens64.json)
# Compare arms paired by seed. One run needs ~21-27 GB, so one run per A6000. The lite and lite_bf16 arms (one 3x3 conv
# in the full-resolution blocks, full-resolution skip added: -0.17 dB brain) ran on the code snapshot at commit 370899b
# of ~/data/seqmri_checkpoints/unet_speed/code; their options are gone.
# CASCADES (default 2) sets the cascade count; runs are named varnet<CASCADES>_r8_<arm>_seed<seed>. At 12 cascades and
# bs 16 base peaks at ~48 GiB, which does not fit an A6000 (the out-of-place LeakyReLU keeps one more activation per conv
# than fastMRI's in-place one); training.compile_unets=true brings it to ~34 GiB (one measured step, not a full run).
# FOREGROUND=1 runs the job in this shell (for scripts/launch/queue_unet_speed_varnet2_r8.sh) instead of detaching it.
#
#   ARM=sens64 GPU=0 SEED=42 scripts/launch/train_unet_speed_varnet2_r8.sh
#   CASCADES=12 ARM=bf16 GPU=0 SEED=42 scripts/launch/train_unet_speed_varnet2_r8.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

ARM="${ARM:?set ARM=base, bf16 or sens64}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
CASCADES="${CASCADES:-2}"
OUT_ROOT="${OUT_ROOT:-$HOME/data/seqmri_checkpoints/unet_speed/runs}"
MASK="${MASK:-$HOME/data/seqmri_checkpoints/unet_speed/static_vd_r8_mask_seed42.pt}"
case "$ARM" in
    base) KNOBS=() ;;
    bf16) KNOBS=(recon.unet_precision=bf16) ;;
    sens64) KNOBS=(sampler.sens_resolution=64) ;;
    *) echo "ARM must be base, bf16 or sens64, got $ARM" >&2; exit 1 ;;
esac

OUT="$OUT_ROOT/varnet${CASCADES}_r8_${ARM}_seed${SEED}"
mkdir -p "$OUT_ROOT"

CMD=(env CUDA_VISIBLE_DEVICES="$GPU" python train.py --model sequential \
    recon=varnet \
    recon.num_cascades="$CASCADES" \
    "${KNOBS[@]}" \
    datasets=brain_knee_balanced \
    datasets.num_workers=12 \
    'sampler.accelerations=[8.0]' \
    sampler.sampler_type=loupe_2d \
    sampler.fixed_mask_path="$MASK" \
    training.batch_size=16 \
    training.lr=0.001 \
    training.lr_scheduler=step \
    training.lr_step_size=40 \
    training.num_epochs=50 \
    training.seed="$SEED" \
    output.out_dir="$OUT" "$@")

if [[ "${FOREGROUND:-0}" == 1 ]]; then
    "${CMD[@]}" > "$OUT.log" 2>&1
else
    source scripts/launch/_launch.sh
    launch_detached "$OUT.log" "${CMD[@]}"
fi
