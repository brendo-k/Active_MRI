#!/usr/bin/env bash
# Does starting LOUPE from a variable-density (VD) prior raise the per-anatomy mask gain (VarNet, R=8, brain + knee)?
#
# The ISMRM 2027 pair (12-cascade VarNet, loupe_2d straight-through, R=8, balanced brain T2 + knee, bs 16, lr 1e-3
# stepped at epoch 90, 100 epochs) gave per-anatomy minus shared, run - run PSNR: knee +0.13 (seed 42) / +0.18 (seed 43),
# brain -0.05 / +0.13 (reports/ismrm2027/anatomy_pair_seed{42,43}.json). Its masks start from uniform logit noise, keep
# that draw's pixel layout, and end near a p=2 radial density (r >= 64 holds 49-54% of the samples, 81% at the start);
# on CG-SENSE the straight-through gradient barely sees the radial density and leaves masks too flat.
# Here every mask row starts at the budget-matched VD prior clamp(c ((D - r) / D)^VD_POWER, 0.01, 0.99), so the two arms
# of a seed start from the same mask, reconstructor, sensitivity net and batch order and differ only in
# sampler.anatomy_conditioning. (The ISMRM pairs also differed in sensitivity-net init and batch order: the per-anatomy
# mask's extra random draw moved the global RNG.)
#   ARM=shared              one mask for every slice
#   ARM=anatomy             one mask per anatomy, chosen by the slice's anatomy label
#   ARM=anatomy_samestart   anatomy, with every anatomy's mask starting from the first one's random draw (the shared
#                           mask's draw at the same seed); with VD_POWER=0 (see the second rule below)
# Seeds 42 and 43. The runs use the repo's current defaults (bf16 cascades, compiled U-Nets, cuDNN autotuner, padded-coil
# skip), so their absolute PSNR is not comparable with the fp32 ISMRM runs; only the per-anatomy - shared deltas are.
#
# Eval, per seed (also the label swap and the mask effect on each reconstructor):
#   python scripts/eval/eval_anatomy_pair_ismrm_r8.py --shared <shared run> --anatcond <anatomy run> --tag vdinit_p4_seed<seed>
#
# The control is the same launcher with VD_POWER=0 (the uniform random start, same code and pairing), seeds 42 and 43,
# queued after the VD_POWER=4 wave; its runs and eval tags read vdinit_p0.
#
# Decision rule, fixed before any result (PSNR, per seed, 95% volume-bootstrap CI; delta = run - run, per-anatomy minus
# shared, from eval_anatomy_pair_ismrm_r8.py):
#   VD start raises the routing gain   knee delta(p4) > knee delta(p0) in both seeds, and brain delta(p4) >= brain
#                                      delta(p0) - 0.05 in both seeds
#   VD start lowers it                 knee delta(p4) < knee delta(p0) in both seeds
#   mixed                              anything else: report per seed, claim nothing
# Also reported: the routed model's own quality, anatomy(p4) minus anatomy(p0) per anatomy (paired per slice, two runs
# on the same val slices), since a larger gap to the shared mask is no help if the routed model itself gets worse; and
# the label swap (knee with the brain mask) for knee specificity. The ISMRM random-start deltas (knee +0.13 / +0.18,
# fp32 code, unpaired sensitivity-net init and batch order) are context only.
# RESULT (2026-10-24, reports/ismrm2027/vdinit_summary.json): the rule passed only because random-start routing collapsed
# (run - run brain -0.25 / -0.69, knee -0.16 / -0.24); VD-start routing gives knee +0.05 / +0.13, brain -0.12 / -0.07, and
# both starts end at the same radial density.
#
# Second question: is it the common start or the smooth one? ARM=anatomy_samestart, VD_POWER=0, seeds 42 and 43, pairs with
# the random-start shared runs (identical recon, sensitivity net, batches and first mask row). Rule, fixed before the runs,
# on run - run deltas versus the shared random-start twin, in both seeds, brain and knee:
#   the common start is enough   samestart delta within 0.05 dB of the VD-start delta, or above it
#   the smooth start is needed   samestart delta within 0.05 dB of the random-start delta, or below it
#   mixed                        anything else
#
#   ARM=anatomy GPU=1 SEED=42 scripts/launch/train_loupe2d_varnet_r8_vdinit_anatomy.sh
# FOREGROUND=1 runs the job in this shell (for scripts/launch/queue_loupe2d_varnet_r8_vdinit_anatomy.sh).
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

ARM="${ARM:?set ARM=shared, anatomy or anatomy_samestart}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
VD_POWER="${VD_POWER:-4}"
OUT_ROOT="${OUT_ROOT:-$HOME/data/seqmri_checkpoints/vdinit_anatomy/runs}"
EXTRA=()
case "$ARM" in
    shared) CONDITIONED=false ;;
    anatomy) CONDITIONED=true ;;
    anatomy_samestart) CONDITIONED=true; EXTRA=(sampler.loupe_same_start=true) ;;
    *) echo "ARM must be shared, anatomy or anatomy_samestart, got $ARM" >&2; exit 1 ;;
esac

OUT="$OUT_ROOT/loupe2d_varnet_r8_brain_knee_${ARM}_100ep_vdinit_p${VD_POWER}_seed${SEED}"
mkdir -p "$OUT_ROOT"

CMD=(env CUDA_VISIBLE_DEVICES="$GPU" python train.py --model sequential \
    sampler.sampler_type=loupe_2d \
    recon=varnet \
    datasets=brain_knee_balanced \
    'sampler.accelerations=[8.0]' \
    training.batch_size=16 \
    training.num_epochs=100 \
    training.lr_step_size=90 \
    sampler.anatomy_conditioning="$CONDITIONED" \
    sampler.loupe_vd_init_power="$VD_POWER" \
    training.seed="$SEED" \
    "${EXTRA[@]}" \
    output.out_dir="$OUT" "$@")

if [[ "${FOREGROUND:-0}" == 1 ]]; then
    "${CMD[@]}" > "$OUT.log" 2>&1
else
    source scripts/launch/_launch.sh
    launch_detached "$OUT.log" "${CMD[@]}"
fi
