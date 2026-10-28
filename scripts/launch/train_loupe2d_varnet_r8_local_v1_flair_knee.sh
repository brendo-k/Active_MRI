#!/usr/bin/env bash
# Shared vs per-condition LOUPE mask on the raw local fastMRI subset v1: brain AXFLAIR + knee PD / PDFS (VarNet, R=8).
#
# Data (configs/datasets/fastmri_local_v1_flair_knee.yaml): tier S train (80 FLAIR volumes, 40 knee PD, 40 knee PDFS),
# manifest val (40 FLAIR, 20 PD, 20 PDFS), raw h5 on /export/scratch. 320x320 grid: readout oversampling cropped in the
# image, knee phase encodes centre-cropped 368 -> 320, all coils kept (zero-padded to 20). FLAIR acquires 198-256 of
# its phase encodes (and 256 readout rows on 512-readout scans), so its k-space only partly fills the grid. The sampler
# confines each slice's budget to its nonzero k-space, so R=8 is of the 320x320 grid: FLAIR is about R=5.6 of what
# it acquires.
#
# Everything else as scripts/launch/train_loupe2d_varnet_r8_vdinit_anatomy.sh: 12-cascade VarNet, loupe_2d
# straight-through, VD p=4 start, bs 16, lr 1e-3 stepped at epoch 90, 100 epochs. The two arms of a seed start from
# the same mask, reconstructor, sensitivity net and batch order, and differ only in sampler.anatomy_conditioning.
#   ARM=shared    one mask for every slice
#   ARM=anatomy   one mask per condition class (brain, knee-PD, knee-PDFS), chosen by the slice's label
#
#   ARM=anatomy GPU=1 SEED=42 scripts/launch/train_loupe2d_varnet_r8_local_v1_flair_knee.sh
# FOREGROUND=1 runs the job in this shell.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

ARM="${ARM:?set ARM=shared or anatomy}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
VD_POWER="${VD_POWER:-4}"
EPOCHS="${EPOCHS:-100}"
OUT_ROOT="${OUT_ROOT:-$HOME/data/seqmri_checkpoints/local_v1_flair_knee/runs}"
case "$ARM" in
    shared) CONDITIONED=false ;;
    anatomy) CONDITIONED=true ;;
    *) echo "ARM must be shared or anatomy, got $ARM" >&2; exit 1 ;;
esac

OUT="$OUT_ROOT/loupe2d_varnet_r8_flair_knee_${ARM}_${EPOCHS}ep_vdinit_p${VD_POWER}_seed${SEED}"
mkdir -p "$OUT_ROOT"

CMD=(env CUDA_VISIBLE_DEVICES="$GPU" python train.py --model sequential \
    sampler.sampler_type=loupe_2d \
    recon=varnet \
    datasets=fastmri_local_v1_flair_knee \
    'misc.resolution=[320,320]' \
    'sampler.accelerations=[8.0]' \
    training.batch_size=16 \
    training.num_epochs="$EPOCHS" \
    training.lr_step_size=90 \
    sampler.anatomy_conditioning="$CONDITIONED" \
    sampler.loupe_vd_init_power="$VD_POWER" \
    training.seed="$SEED" \
    output.out_dir="$OUT" "$@")

if [[ "${FOREGROUND:-0}" == 1 ]]; then
    "${CMD[@]}" > "$OUT.log" 2>&1
else
    source scripts/launch/_launch.sh
    launch_detached "$OUT.log" "${CMD[@]}"
fi
