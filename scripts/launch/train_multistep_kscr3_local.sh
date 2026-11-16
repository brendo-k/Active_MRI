#!/bin/bash
# kscr3 of exp/multistep_adaptivity (reports/multistep_adaptivity/addendum_2026-11-16.md) on the lab server: the
# SeqMRI-faithful arm on GPU 0, and one-step LOUPE then its per-condition version on GPU 1, each scored on the dev half
# when it finishes. Run from the worktree root:
#
#   bash scripts/launch/train_multistep_kscr3_local.sh
#
# SMOKE=1 trains each arm for 10% of one epoch and scores 16 slices, without wandb; point OUT and SCORES at scratch.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON=.venv/bin/python
OUT="${OUT:-$HOME/data/seqmri_checkpoints/multistep_kscr3}"
SCORES="${SCORES:-reports/multistep_adaptivity/kscr3/scores}"
ARMS=(F_4_s42_kscr3 L_1_s42_kscr3 Lc_1_s42_kscr3)
QUEUES=("F_4_s42_kscr3" "L_1_s42_kscr3 Lc_1_s42_kscr3")  # one queue per GPU, each run in order
GPUS=(0 1)

EPOCHS=100
EXTRA=()
SCORE_ARGS=(--gradients)
if [[ "${SMOKE:-0}" == 1 ]]; then
    EPOCHS=1
    EXTRA=(training.limit_batches=0.1)
    SCORE_ARGS=(--limit 16 --no-wandb --figures 0)
    export WANDB_MODE=disabled
fi
RECIPE=(--recon varnet --dataset fastmri_local_v1_knee --acceleration 8 --batch-size 16 --seed 42
        'misc.resolution=[320,320]' sampler.budget_over=acquired sampler.preselect_num=40 recon.num_cascades=4 recon.checkpoint_cascades=true "training.num_epochs=$EPOCHS"
        training.lr_step_size=90
        datasets.val_half=dev datasets.train_slice_stride=5
        datasets.roots.local_batch2=/export/scratch/bkadota/fastmri_raw/knee "${EXTRA[@]}")

arm_args() {
    case "$1" in
        F_4_s42_kscr3) echo "--model sequential sampler.sampler_type=seqmri_2d training.num_step=4 sampler.anatomy_conditioning=false" ;;
        L_1_s42_kscr3) echo "--model loupe training.num_step=1 sampler.anatomy_conditioning=false" ;;
        Lc_1_s42_kscr3) echo "--model loupe training.num_step=1 sampler.anatomy_conditioning=true" ;;
        *) echo "unknown arm $1" >&2; return 1 ;;
    esac
}

run_arm() {
    local arm=$1 gpu=$2 args
    read -r -a args <<< "$(arm_args "$arm")"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON train.py "${args[@]}" "${RECIPE[@]}" "output.out_dir=$OUT/$arm" &&
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON scripts/eval/eval_multistep_adaptivity.py score --run "$OUT/$arm" \
            --name "$arm" --out "$SCORES" "${SCORE_ARGS[@]}"
}

run_queue() {
    local gpu=$1 arm
    shift
    for arm in "$@"; do
        if run_arm "$arm" "$gpu" > "$OUT/$arm.log" 2>&1; then echo "$arm exit 0"; else echo "$arm exit $?"; fi
    done
}

for arm in "${ARMS[@]}"; do
    if [[ -e "$OUT/$arm/last_model.pt" ]]; then
        echo "$OUT/$arm already holds a run; move it aside to rerun (crashed runs are rerun in full)" >&2
        exit 1
    fi
done
mkdir -p "$OUT" "$SCORES"
pids=()
for i in "${!QUEUES[@]}"; do
    read -r -a queue <<< "${QUEUES[$i]}"
    run_queue "${GPUS[$i]}" "${queue[@]}" &
    pids+=($!)
done
wait "${pids[@]}"
