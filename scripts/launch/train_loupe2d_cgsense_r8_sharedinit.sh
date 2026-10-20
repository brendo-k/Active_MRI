#!/usr/bin/env bash
# Why does the CG-SENSE (R=8) per-anatomy knee mask lose to the shared mask (-0.25 dB, worse on both anatomies)?
# Start the knee row from the trained shared mask instead of a fresh random draw, and see whether it holds.
#
# Both arms continue the shared 100-epoch run (its epoch-99 mask and sensitivity net, fresh Adam) for EPOCHS more
# epochs at that run's constant lr 1e-3, seed 42:
#   test     one mask per anatomy, both rows starting from the shared mask
#   control  the shared mask, continued
# Score with `scripts/diagnostics/anatcond_cgsense_cross_eval_r8.py --pair sharedinit` (control vs test, plus the start
# map). The knee effect is anat_own - shared mask.
#
# Decision rule, fixed before the runs:
#   init/basin was the problem  knee effect CI covers 0 or is positive, and the knee row's correlation with the start
#                               and its r >= 64 share track the brain row and the continued shared mask
#   knee gradient pulls it away knee effect negative, and the knee row decorrelates from the start or flattens more than
#                               both the brain row and the continued shared mask (check whether knee-PDFS carries it)
#   ambiguous                   all three maps drift about equally and the knee effect is small: EPOCHS was too short
#                               to separate them
set -euo pipefail

cd "$(dirname "$0")/../.."

SOURCE="$(pwd)/checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_const_100ep/last_model.pt"
TEST_GPU="${TEST_GPU:-1}"
CONTROL_GPU="${CONTROL_GPU:-3}"
EPOCHS="${EPOCHS:-50}"

common=(
    sampler.sens_upsample_method=bilinear
    training.lr_scheduler=constant
    training.num_epochs="$EPOCHS"
    sampler.init_mask_checkpoint="$SOURCE"
    output.pretrained_recon="$SOURCE"
)

GPU="$TEST_GPU" ANATOMY_MASKS=1 OUT=checkpoints/loupe2d_cgsense_r8_brain_knee_anatcond_sharedinit \
    scripts/launch/train_loupe2d_cgsense_r8_brain_knee.sh "${common[@]}" \
    output.run_name=loupe2d_R8_cg_sense_anatcond_sharedinit_brain+knee
GPU="$CONTROL_GPU" ANATOMY_MASKS=0 OUT=checkpoints/loupe2d_cgsense_r8_brain_knee_shared_cont \
    scripts/launch/train_loupe2d_cgsense_r8_brain_knee.sh "${common[@]}" \
    output.run_name=loupe2d_R8_cg_sense_shared_cont_brain+knee
