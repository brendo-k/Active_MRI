#!/usr/bin/env bash
# Can a sampler predict a patient-specific mask? Input-adaptive conv_2d vs a static shared LOUPE control, VarNet, R=8 of
# each slice's acquired k-space, on the raw local fastMRI subset v1: brain AXFLAIR + knee PD / PDFS (2026-10-28).
#
# What changed since the conv_2d runs of 2026-10-15/16 (reports/conv_vs_loupe_r8):
#   - the conv sampler's input is the RSS of the observed coil images (sampler.coil_combine=rss). It was sum_c S_c IFFT(k_c),
#     30-70% off the target in magnitude (|sum S^2| ~ 0.4-0.7), and even the SENSE combination with conj(S_c) has a
#     pixel-to-pixel random phase: the RSS L1 loss leaves the maps' common phase free, and the jointly trained
#     sensitivity net makes it random (39% of neighbour pairs jump > pi/2), so its real/imag channels were noise
#   - R counts each slice's acquired k-space (sampler.budget_over=acquired): FLAIR acquires ~64% of the 320 grid, so a
#     grid budget had run it at R~5 of what it acquires
#   - the output is the RSS of the coil k-space x S_c zero-filled outside the acquired extent, like the target
#
# Arms (everything else as scripts/launch/train_loupe2d_varnet_r8_local_v1_flair_knee.sh: 12-cascade VarNet, bs 16,
# lr 1e-3 stepped at epoch 90, 100 epochs; the reconstructor, sensitivity net and batch order start the same per seed):
#   ARM=conv    conv_2d, image input (the RSS of the zero-filled centre's coil images), straight-through, mask and STE
#               gradient floors 0.05, as the earlier conv runs
#   ARM=loupe   one shared loupe_2d mask from the VD p=4 start, as the local v1 LOUPE runs
#   ARM=conv_encmlp  ARM=conv plus sampler.head=encoder_mlp (2026-10-28): an MLP on the U-Net bottleneck gives every
#               k-space location its own input-dependent logit, zero-mean over the batch, so no static per-location
#               bias. The head is built on its own RNG stream: at seed 42 every other weight and the RNG stream equal
#               ARM=conv's (checked), so it is one change against the finished conv run
#
# Readout, fixed before the runs finish (scripts/eval/eval_patient_specific_r8.py):
#   Headline: the conv run's other-volume swap cost per condition (own mask minus the mask the sampler made for another
#   volume of the same condition, both re-rescaled to the slice's own budget), PSNR, 95% volume-bootstrap CI.
#   "Predicts a patient-specific mask" = cost > 0 with the CI excluding 0 in a condition. The LOUPE control's swap cost
#   is 0 by construction (checked on the local v1 LOUPE pair). Expectation from the earlier ACS analyses: <= ~0.06 dB per
#   slice. Secondary: conv own minus LOUPE own, which also differs in recipe and has one seed (seed spread 0.3-0.5 dB).
#
# Readout for ARM=conv_encmlp, fixed before it starts. The conv run passed the swap-cost rule in knee (+0.05 dB) while
# its own mask was no better than its condition-mean mask (+0.01 / +0.005 / -0.03 dB), so swap cost is not the headline:
#   Headline: own mask minus the condition-mean mask (adaptive_gain_over_condition_mean), PSNR, 95% volume-bootstrap CI,
#   per condition. "Predicts a patient-specific mask" = > 0 with the CI excluding 0 in a condition.
#   Secondary: slice-vs-condition-mean mask correlation below the conv run's 0.99 (the head is used at all); own-mask
#   PSNR minus the conv run's, paired per slice from the two eval .pt files; own minus LOUPE own.
#   Expectation: the sampler still reads only the 10x10 centre; the earlier ACS analyses cap a per-slice mask at ~0.06 dB.
#
#   ARM=conv GPU=0 SEED=42 scripts/launch/train_patient_specific_r8_local_v1_flair_knee.sh
# FOREGROUND=1 runs the job in this shell. Extra arguments are Hydra overrides (e.g. recon.checkpoint_cascades=true).
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

ARM="${ARM:?set ARM=conv, conv_encmlp or loupe}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
OUT_ROOT="${OUT_ROOT:-$HOME/data/seqmri_checkpoints/patient_specific_r8_local_v1/runs}"
CONV=(sampler.sampler_type=conv_2d sampler.mask_grad_floor=0.05 sampler.ste_grad_floor=0.05 sampler.coil_combine=rss)
case "$ARM" in
    conv) SAMPLER=("${CONV[@]}") ;;
    conv_encmlp) SAMPLER=("${CONV[@]}" sampler.head=encoder_mlp) ;;
    loupe) SAMPLER=(sampler.sampler_type=loupe_2d sampler.loupe_vd_init_power=4 sampler.anatomy_conditioning=false) ;;
    *) echo "ARM must be conv, conv_encmlp or loupe, got $ARM" >&2; exit 1 ;;
esac

OUT="$OUT_ROOT/patient_specific_${ARM}_r8_flair_knee_${EPOCHS}ep_seed${SEED}"
mkdir -p "$OUT_ROOT"

CMD=(env CUDA_VISIBLE_DEVICES="$GPU" python train.py --model sequential \
    recon=varnet \
    datasets=fastmri_local_v1_flair_knee \
    'misc.resolution=[320,320]' \
    'sampler.accelerations=[8.0]' \
    sampler.budget_over=acquired \
    training.num_step=1 \
    training.batch_size=16 \
    training.num_epochs="$EPOCHS" \
    training.lr_step_size=90 \
    training.seed="$SEED" \
    "${SAMPLER[@]}" \
    output.out_dir="$OUT" "$@")

if [[ "${FOREGROUND:-0}" == 1 ]]; then
    "${CMD[@]}" > "$OUT.log" 2>&1
else
    source scripts/launch/_launch.sh
    launch_detached "$OUT.log" "${CMD[@]}"
fi
