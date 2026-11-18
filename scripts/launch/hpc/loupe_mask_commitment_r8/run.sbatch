#!/bin/bash
# LOUPE mask commitment at R=8 (reports/loupe_mask_commitment_r8/proposal.md): the 12-cascade lr 1e-3 step control of
# varnet_loupe_lr_schedule_r8 at full local v1 FLAIR + knee, 100 epochs, and one change per arm. Task = 2 * arm_index + seed_index.
#
# Layout under $EXP_DIR: code/ (git archive of the run-state commit), runs/<arm>_s<seed>/, slurm_log/.
#
#   cd ~/scratch/loupe_mask_commitment_r8
#   sbatch --array=0-7 --export=ALL,SMOKE=1 --time=00:40:00 code/scripts/launch/hpc/loupe_mask_commitment_r8/run.sbatch
#   sbatch --array=0-7 code/scripts/launch/hpc/loupe_mask_commitment_r8/run.sbatch
#
#SBATCH --job-name=lmc_r8
#SBATCH --partition=gpu
#SBATCH --account=hpc_core
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96GB
#SBATCH --time=07:00:00
#SBATCH --output=slurm_log/%x-%A_%a.log

set -eo pipefail

: "${SLURM_ARRAY_TASK_ID:?submit with --array}"
EXP_DIR="${EXP_DIR:-$HOME/scratch/loupe_mask_commitment_r8}"
SEEDS=(42 43)
ARMS=(control clamp anneal mix)
ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 2))]:?task $SLURM_ARRAY_TASK_ID has no arm (valid: 0-$((2 * ${#ARMS[@]} - 1)))}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}"

case "$ARM" in
    control) ARM_OVERRIDES=() ;;
    clamp) ARM_OVERRIDES=(sampler.loupe_logit_limit=6.0 sampler.logit_leak=0.01) ;;
    anneal) ARM_OVERRIDES=(sampler.slope_start=2.0 sampler.slope_anneal_epochs=20) ;;
    mix) ARM_OVERRIDES=(sampler.flat_mask_fraction=0.25) ;;
esac

EPOCHS=100
VAL_EVERY=5
RUNS_DIR=runs
if [[ -n "$SMOKE" ]]; then
    EPOCHS=2
    VAL_EVERY=1
    RUNS_DIR=smoke
    ARM_OVERRIDES+=(training.limit_batches=0.02 datasets.val_limit_volumes=0.05)
fi

NAME="${ARM}_s${SEED}"
OUT_DIR="$EXP_DIR/$RUNS_DIR/$NAME"
if [[ -e "$OUT_DIR/last_model.pt" ]]; then
    mv "$OUT_DIR" "$OUT_DIR.attempt-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUT_DIR"

export WANDB_MODE=offline
export WANDB_RUN_GROUP=loupe_mask_commitment_r8_$RUNS_DIR
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /mnt/software/ohpc/pub/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate dl

echo "arm=$ARM seed=$SEED epochs=$EPOCHS task=$SLURM_ARRAY_TASK_ID host=$(hostname) out=$OUT_DIR overrides=${ARM_OVERRIDES[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$EXP_DIR/code"
python train.py --model sequential recon=varnet recon.num_cascades=12 \
    datasets=fastmri_local_v1_flair_knee \
    "datasets.roots.hpc=\"$HOME/scratch/data/fastmri/{anatomy}/data\"" \
    "datasets.roots.local_batch2=$HOME/scratch/data/fastmri/knee/data" \
    datasets.limit_volumes=1 \
    'misc.resolution=[320,320]' \
    'sampler.accelerations=[8.0]' sampler.budget_over=acquired \
    sampler.sampler_type=loupe_2d sampler.loupe_vd_init_power=0 sampler.anatomy_conditioning=false \
    sampler.lr=0.001 \
    training.num_step=1 training.batch_size=16 training.num_epochs="$EPOCHS" \
    training.lr=0.001 training.lr_scheduler=step training.lr_step_size=90 training.lr_gamma=0.1 \
    training.val_every="$VAL_EVERY" training.seed="$SEED" \
    "${ARM_OVERRIDES[@]}" \
    "output.out_dir=$OUT_DIR" "output.run_name=lmc_$NAME"
