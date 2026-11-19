#!/bin/bash
# Score one checkpoint of a multistep_adaptivity run with the K-step eval, e.g. the pilot gate on epoch_010.pt
# (proposal section 8, P1). train.sbatch already scores each run's last_model.pt.
#
#   sbatch --gres=gpu:<pinned type>:1 --export=ALL,ARM=A_16_s42,CHECKPOINT=epoch_010.pt,GPU_NAME=<pinned name> \
#       code/scripts/launch/hpc/multistep_adaptivity/eval.sbatch
#
#SBATCH --job-name=multistep_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --output=slurm_log/%x-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brenden.kadota@ucsf.edu

set -eo pipefail
source "${EXP_DIR:-$HOME/scratch/multistep_adaptivity}/code/scripts/launch/hpc/multistep_adaptivity/common.sh"
: "${ARM:?submit with --export=ALL,ARM=<arm>,GPU_NAME=<card>}"
CHECKPOINT="${CHECKPOINT:-last_model.pt}"
NAME="${NAME:-${ARM}$([[ "$CHECKPOINT" == last_model.pt ]] || echo "_${CHECKPOINT%.pt}")}"
start_job

EVAL_ARGS=()
[[ "$ARM" == B_* ]] && EVAL_ARGS+=(--no-sensitivity)
python scripts/eval/eval_multistep_adaptivity.py score --run "$RUNS_DIR/$ARM" --checkpoint "$CHECKPOINT" --name "$NAME" \
    --out "$SCORES_DIR" --gradients "${EVAL_ARGS[@]}"
