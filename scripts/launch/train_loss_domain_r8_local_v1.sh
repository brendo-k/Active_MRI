#!/usr/bin/env bash
# Loss domain at R=8 (reports/loss_domain/proposal.md, docs/experiments/loss_domain): one arm and seed of the shared
# LOUPE + VarNet recipe on the raw local fastMRI subset v1 (brain AXFLAIR + knee PD / PDFS). The arms differ only in the
# training loss; every arm is scored on the RSS output under the apodized metric mask.
#
#   ARM=rss           production L1 of the RSS image (the twin of every other arm)
#   ARM=complex       mean |x_E - x|: the extent-projected coil-combined image against the cached ESPIRiT image x
#   ARM=kspace_sense  mean |F(S x) E - F(S.detach() x_espirit) E| over coils: coil k-space through the learned maps
#   ARM=kspace_rss    mean |F(rss) - F(target)|: the RSS images in k-space, phase-blind
#   ARM=kspace_rss_M  kspace_rss with both RSS images multiplied by M = |x_espirit| > 0
#
# Shared: loupe_2d (VD p=4 start, no conditioning), R=8 of each slice's acquired k-space, 12-cascade VarNet, bs 16,
# 100 epochs, lr 1e-3 stepped at epoch 90, seed-paired (same init, batch order and mask-draw stream per seed).
#
#   ARM=complex GPU=0 SEED=42 scripts/launch/train_loss_domain_r8_local_v1.sh
# FOREGROUND=1 runs the job in this shell. Extra arguments are Hydra overrides (e.g. recon.checkpoint_cascades=true, or
# the CoreHPC data roots). COMPLEX_CACHE points at the ESPIRiT image cache.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

ARM="${ARM:?set ARM=rss, complex, kspace_sense, kspace_rss or kspace_rss_M}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
OUT_ROOT="${OUT_ROOT:-$HOME/data/seqmri_checkpoints/loss_domain/runs}"
COMPLEX_CACHE="${COMPLEX_CACHE:-/export/scratch/bkadota/fastmri_complex_espirit}"
CACHE=("datasets.complex_cache_dir=$COMPLEX_CACHE")
case "$ARM" in
    rss) LOSS=(loss.loss_type=l1) ;;
    complex) LOSS=(loss.loss_type=complex_l1 "${CACHE[@]}") ;;
    kspace_sense) LOSS=(loss.loss_type=kspace_sense_l1 "${CACHE[@]}") ;;
    kspace_rss) LOSS=(loss.loss_type=kspace_rss_l1) ;;
    kspace_rss_M) LOSS=(loss.loss_type=kspace_rss_l1 loss.support_mask=true "${CACHE[@]}") ;;
    *) echo "ARM must be rss, complex, kspace_sense, kspace_rss or kspace_rss_M, got $ARM" >&2; exit 1 ;;
esac

OUT="$OUT_ROOT/loss_domain_${ARM}_r8_flair_knee_${EPOCHS}ep_seed${SEED}"
mkdir -p "$OUT_ROOT"

CMD=(env CUDA_VISIBLE_DEVICES="$GPU" python train.py --model sequential \
    recon=varnet \
    datasets=fastmri_local_v1_flair_knee \
    'misc.resolution=[320,320]' \
    'sampler.accelerations=[8.0]' \
    sampler.budget_over=acquired \
    sampler.sampler_type=loupe_2d \
    sampler.loupe_vd_init_power=4 \
    sampler.anatomy_conditioning=false \
    sampler.mask_train_mode=straight_through \
    sampler.coil_combine=sense \
    recon.csme=false \
    training.num_step=1 \
    training.batch_size=16 \
    training.num_epochs="$EPOCHS" \
    training.lr_step_size=90 \
    training.seed="$SEED" \
    "${LOSS[@]}" \
    output.out_dir="$OUT" "$@")

if [[ "${FOREGROUND:-0}" == 1 ]]; then
    "${CMD[@]}" > "$OUT.log" 2>&1
else
    source scripts/launch/_launch.sh
    launch_detached "$OUT.log" "${CMD[@]}"
fi
