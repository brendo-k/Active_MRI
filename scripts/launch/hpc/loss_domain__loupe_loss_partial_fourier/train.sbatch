#!/bin/bash
# One loss_domain__loupe_loss_partial_fourier run on CoreHPC, then its score on the same card. Arm names are in
# common.sh. Submit from $EXP_DIR (~/scratch/loss_domain__loupe_loss_partial_fourier):
#
#   sbatch --gres=gpu:nvidia_rtx_pro_6000:1 --export=ALL,ARM=Lcplx_1_s42_tF,GPU_NAME='RTX PRO 6000' \
#       code/scripts/launch/hpc/loss_domain__loupe_loss_partial_fourier/train.sbatch
#
#SBATCH --job-name=loupe_lpf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_log/%x-%j.log

set -eo pipefail
TOPIC=loss_domain__loupe_loss_partial_fourier
source "${EXP_DIR:-$HOME/scratch/$TOPIC}/code/scripts/launch/hpc/$TOPIC/common.sh"
: "${ARM:?submit with --export=ALL,ARM=<arm>,GPU_NAME=<card>}"
start_job
loss_pf_arm_args "$ARM"

RUN_DIR="$RUNS_DIR/$ARM"
if [[ -e "$RUN_DIR/last_model.pt" ]]; then
    echo "$RUN_DIR already holds a run; move it aside to rerun (crashed runs are rerun in full)" >&2
    exit 1
fi
echo "arm=$ARM run=$RUN_DIR"
python train.py "${ARM_ARGS[@]}" "${RECIPE[@]}" "output.out_dir=$RUN_DIR"
python scripts/eval/eval_multistep_adaptivity.py score --run "$RUN_DIR" --name "$ARM" --out "$SCORES_DIR" --gradients
