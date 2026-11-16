#!/bin/bash
# Follow-up of the lr x schedule sweep at full local v1 FLAIR + knee, 100 epochs
# (reports/varnet_loupe_lr_schedule_r8/followup_proposal.md). Task = 2 * arm_index + seed_index.
#
#   cd ~/scratch/varnet_loupe_lr_schedule_r8
#   sbatch --array=0-9 code/scripts/launch/hpc/varnet_loupe_lr_schedule_r8/followup.sbatch
#   CASCADES=6 sbatch --array=0-5 --time=04:00:00 code/scripts/launch/hpc/varnet_loupe_lr_schedule_r8/followup.sbatch   # followup2
#
#SBATCH --job-name=vlls_full
#SBATCH --partition=gpu
#SBATCH --account=hpc_core
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96GB
#SBATCH --time=07:00:00
#SBATCH --output=slurm_log/%x-%A_%a.log

set -eo pipefail

: "${SLURM_ARRAY_TASK_ID:?submit with --array}"
EXP_DIR="${EXP_DIR:-$HOME/scratch/varnet_loupe_lr_schedule_r8}"
SEEDS=(42 43)
CASCADES="${CASCADES:-12}"
RUNS_DIR=runs_full
RUN_PREFIX=vlls_full
if [[ "$CASCADES" != 12 ]]; then RUNS_DIR=runs_full_c$CASCADES; RUN_PREFIX=vlls_full_c$CASCADES; fi
ARMS=(lr1e3_step lr2e3_step lr4e3_step lr2e3_wcos lr4e3_wcos)
ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 2))]:?task $SLURM_ARRAY_TASK_ID has no arm (valid: 0-$((2 * ${#ARMS[@]} - 1)))}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}"

case "$ARM" in
    lr1e3_*) LR=0.001 ;;
    lr2e3_*) LR=0.002 ;;
    lr4e3_*) LR=0.004 ;;
esac
case "$ARM" in
    *_step) SCHEDULE=(training.lr_scheduler=step training.lr_step_size=90 training.lr_gamma=0.1) ;;
    *_wcos) SCHEDULE=(training.lr_scheduler=warmup_cosine training.lr_warmup_epochs=2) ;;
esac

NAME="${ARM}_s${SEED}"
OUT_DIR="$EXP_DIR/$RUNS_DIR/$NAME"
if [[ -e "$OUT_DIR/last_model.pt" ]]; then
    mv "$OUT_DIR" "$OUT_DIR.attempt-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUT_DIR"

export WANDB_MODE=offline
export WANDB_RUN_GROUP=varnet_loupe_lr_schedule_r8_$RUNS_DIR
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /mnt/software/ohpc/pub/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate dl

echo "arm=$ARM seed=$SEED lr=$LR cascades=$CASCADES task=$SLURM_ARRAY_TASK_ID host=$(hostname) out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$EXP_DIR/code"
python train.py --model sequential recon=varnet recon.num_cascades="$CASCADES" \
    datasets=fastmri_local_v1_flair_knee \
    "datasets.roots.hpc=\"$HOME/scratch/data/fastmri/{anatomy}/data\"" \
    "datasets.roots.local_batch2=$HOME/scratch/data/fastmri/knee/data" \
    datasets.limit_volumes=1 \
    'misc.resolution=[320,320]' \
    'sampler.accelerations=[8.0]' sampler.budget_over=acquired \
    sampler.sampler_type=loupe_2d sampler.loupe_vd_init_power=0 sampler.anatomy_conditioning=false \
    sampler.lr=0.001 \
    training.num_step=1 training.batch_size=16 training.num_epochs=100 \
    training.lr="$LR" "${SCHEDULE[@]}" \
    training.val_every=5 training.seed="$SEED" \
    "output.out_dir=$OUT_DIR" "output.run_name=${RUN_PREFIX}_$NAME"
