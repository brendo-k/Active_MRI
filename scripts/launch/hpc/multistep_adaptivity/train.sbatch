#!/bin/bash
# One multistep_adaptivity run on CoreHPC (proposal section 4.4), then its eval on the same card (section 5).
# ARM names are listed above arm_args in common.sh. Submit from $EXP_DIR (see run.md):
#
#   sbatch --gres=gpu:<pinned type>:1 --export=ALL,ARM=B_42,GPU_NAME=<pinned name> code/scripts/launch/hpc/multistep_adaptivity/train.sbatch
#
#SBATCH --job-name=multistep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_log/%x-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brenden.kadota@ucsf.edu

set -eo pipefail
source "${EXP_DIR:-$HOME/scratch/multistep_adaptivity}/code/scripts/launch/hpc/multistep_adaptivity/common.sh"
: "${ARM:?submit with --export=ALL,ARM=<arm>,GPU_NAME=<card>}"
start_job
arm_args "$ARM"

RUN_DIR="$RUNS_DIR/$ARM"
if [[ -e "$RUN_DIR/last_model.pt" ]]; then
    echo "$RUN_DIR already holds a run; move it aside to rerun (crashed runs are rerun in full)" >&2
    exit 1
fi
echo "arm=$ARM run=$RUN_DIR"
python train.py "${ARM_ARGS[@]}" "${RECIPE[@]}" "output.out_dir=$RUN_DIR"

EVAL_ARGS=()
[[ "$ARM" == B_* ]] && EVAL_ARGS+=(--no-sensitivity)
python scripts/eval/eval_multistep_adaptivity.py score --run "$RUN_DIR" --name "$ARM" --out "$SCORES_DIR" \
    --gradients "${EVAL_ARGS[@]}"
