#!/usr/bin/env bash
# Regenerates every result file in reports/muon_layer_speed/ from the finished runs in
# ~/data/seqmri_checkpoints/muon_layer_speed (untracked; names, overrides and wandb ids in manifest.json).
# Needs one idle GPU and wandb access. The eval reuses eval.per_slice.npz: delete it to rescore every checkpoint.
#
# The training runs came from scripts/launch/varnet_train_speed_worker.sh. queue_jobs.txt holds their job lines and
# queue_rolling_eval.sh the loop that scored them. BEFORE and AFTER are git worktrees of the two code snapshots the runs used:
#   git worktree add ../run-mls-0920f8b 0920f8b && git worktree add ../run-mls-e61a166 e61a166 (each with the .venv symlink)
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=reports/muon_layer_speed
QUEUE=~/data/seqmri_checkpoints/muon_layer_speed
REFERENCE_QUEUE=~/data/seqmri_checkpoints/muon_layernorm_100ep
BEFORE=$(realpath ../run-mls-0920f8b)
AFTER=$(realpath ../run-mls-e61a166)
WORK=${TMPDIR:-/tmp}/muon_layer_speed
RUNS="ref_s42 e30_cos_s42 e30_cos_ema_s42 e30_step_s42 e30_step_ema_s42 e30_bs8_s42 e30_bs8_ema_s42 e30_cos_muon2e3_s42 e30_cos_muon2e3_ema_s42 e30_step_c16_s42 e30_step_c16_ema_s42 ref_maskof_e30_step_s42 e30_step_maskof_ref_s42"
TRAINED="ref_s42 e30_cos_s42 e30_step_s42 e30_bs8_s42 e30_cos_muon2e3_s42 e30_step_c16_s42"

# Exactness of the step-time changes
python scripts/diagnostics/muon_layer_speed_fused_check.py $OUT/fused_check_full_scale.json
python scripts/diagnostics/muon_layer_speed_gradcheck.py $OUT/gradcheck.json
MLS_NO_TF32=1 python scripts/diagnostics/muon_layer_speed_gradcheck.py $OUT/gradcheck_fp32.json recon.unet_precision=fp32 sampler.sens_precision=fp32 training.batch_size=6
MLS_SMOOTH_LOSS=1 MLS_NO_TF32=1 python scripts/diagnostics/muon_layer_speed_gradcheck.py $OUT/gradcheck_fp32_smooth.json recon.unet_precision=fp32 sampler.sens_precision=fp32 training.batch_size=6
python scripts/diagnostics/muon_layer_speed_equivalence.py --before "$BEFORE" --after "$AFTER" --work-dir "$WORK/equivalence" --out $OUT/equivalence.json

# Step time on an idle GPU: the benchmark decides the criteria
python scripts/diagnostics/muon_layer_speed_bench.py --out $OUT/bench.json --before "$BEFORE" --after "$AFTER" --work-dir "$WORK/bench"

# The profiles show where the time goes. Each came from the code of its stage, not from one commit, so regenerating them means
# checking that code out: control and channels_last at 0920f8b's step code, cheap at 13f768d, the fused stages at 81e3d67. The
# layer_bf16 profiles came from the class in layer_bf16_reverted.txt, which was never committed. profile_fused_pinned.json is the
# fused stage NUMA-pinned, from before the reference-cycle fix, and has no command here.
python scripts/diagnostics/muon_layer_speed_profile.py $OUT/profile_control.json
python scripts/diagnostics/muon_layer_speed_profile.py $OUT/profile_channels_last.json training.channels_last_unets=true
python scripts/diagnostics/muon_layer_speed_profile.py $OUT/profile_cheap.json training.fused_adam=true
python scripts/diagnostics/muon_layer_speed_profile.py $OUT/profile_fused_backward.json training.fused_adam=true
MLS_GC_FREEZE=1 python scripts/diagnostics/muon_layer_speed_profile.py $OUT/profile_gc_freeze.json training.fused_adam=true
MLS_GC_FREEZE=1 python scripts/diagnostics/muon_layer_speed_profile.py $OUT/profile_group_senscl.json training.fused_adam=true training.channels_last_unets=sens
MLS_GC_FREEZE=1 python scripts/diagnostics/muon_layer_speed_profile.py $OUT/profile_chans16_nhwc.json training.fused_adam=true recon.chans=16 training.channels_last_unets=true

# Paired scores of every checkpoint against the same-seed reference, from the snapshot the queue scored with
python scripts/diagnostics/muon_layer_speed_swap_masks.py "$QUEUE" e30_step ref 42
(cd "$BEFORE" && PYTHONPATH=$PWD python scripts/eval/varnet_train_speed_eval.py --queue-dir "$QUEUE" --runs $RUNS --control ref --out "$OLDPWD/$OUT/eval.json")

# Validation curves and epoch times, masks, tables and the figure
python scripts/eval/varnet_train_speed_curves.py --queue-dir "$QUEUE" --prefix mls_ --out $OUT/curves.json
python scripts/eval/varnet_train_speed_curves.py --queue-dir "$REFERENCE_QUEUE" --prefix ml100_ --out $OUT/curves_reference.json
python scripts/diagnostics/varnet_train_speed_masks.py $OUT/masks.json --queue-dir "$QUEUE" --control ref --runs $TRAINED
python scripts/eval/muon_layer_speed_summary.py $OUT --queue-dir "$QUEUE"
python scripts/plots/muon_layer_speed_curves.py $OUT/curves.json $OUT/curves_reference.json $OUT/figures/curves.png
