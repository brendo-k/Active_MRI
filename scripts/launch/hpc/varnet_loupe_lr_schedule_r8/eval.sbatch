#!/bin/bash
# Full-val paired scoring of the varnet_loupe_lr_schedule_r8 runs.
#   sbatch --export=ALL,RUNS=runs,CKPT=last_model.pt code/scripts/launch/hpc/varnet_loupe_lr_schedule_r8/eval.sbatch
#SBATCH --job-name=vlls_eval
#SBATCH --partition=gpu
#SBATCH --account=hpc_core
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96GB
#SBATCH --time=06:00:00
#SBATCH --output=slurm_log/%x-%j.log
set -eo pipefail
EXP_DIR="${EXP_DIR:-$HOME/scratch/varnet_loupe_lr_schedule_r8}"
RUNS="${RUNS:?runs or runs_full}"
CKPT="${CKPT:-last_model.pt}"
export WANDB_MODE=offline HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1
source /mnt/software/ohpc/pub/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate dl
cd "$EXP_DIR/code"
python scripts/eval/varnet_loupe_lr_schedule_r8_eval.py --runs-dir "$EXP_DIR/$RUNS" --checkpoint "$CKPT" \
    --out "$EXP_DIR/eval/${RUNS}_${CKPT%.pt}.json"
