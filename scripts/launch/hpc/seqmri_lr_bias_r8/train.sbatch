#!/bin/bash
# One seqmri_lr_bias_r8 run on CoreHPC, then its score on the same card. Arm names are in common.sh. Submit from
# $EXP_DIR (~/scratch/seqmri_lr_bias_r8):
#
#   sbatch --gres=gpu:nvidia_rtx_pro_6000:1 --export=ALL,ARM=F_4_s42_lrs_tF,GPU_NAME='RTX PRO 6000' code/scripts/launch/hpc/seqmri_lr_bias_r8/train.sbatch
#
#SBATCH --job-name=seqmri_lrb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_log/%x-%j.log

set -eo pipefail
source "${EXP_DIR:-$HOME/scratch/seqmri_lr_bias_r8}/code/scripts/launch/hpc/seqmri_lr_bias_r8/common.sh"
: "${ARM:?submit with --export=ALL,ARM=<arm>,GPU_NAME=<card>}"
start_job
lr_bias_arm_args "$ARM"

RUN_DIR="$RUNS_DIR/$ARM"
if [[ -e "$RUN_DIR/last_model.pt" ]]; then
    echo "$RUN_DIR already holds a run; move it aside to rerun (crashed runs are rerun in full)" >&2
    exit 1
fi
echo "arm=$ARM run=$RUN_DIR"
python train.py "${ARM_ARGS[@]}" "${RECIPE[@]}" "output.out_dir=$RUN_DIR"
python scripts/eval/eval_multistep_adaptivity.py score --run "$RUN_DIR" --name "$ARM" --out "$SCORES_DIR" --gradients
