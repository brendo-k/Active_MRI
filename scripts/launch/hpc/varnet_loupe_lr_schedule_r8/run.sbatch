#!/bin/bash
# lr x schedule sweep for joint VarNet + LOUPE at R=8 on 20% of local v1 FLAIR + knee, 400 epochs
# (reports/varnet_loupe_lr_schedule_r8/proposal.md). One array task per (arm, seed): task = 3 * arm_index + seed_index.
#
# Layout under $EXP_DIR: code/ (git archive of the run-state commit), runs/<arm>_s<seed>/, slurm_log/.
#
#   cd ~/scratch/varnet_loupe_lr_schedule_r8
#   sbatch --array=0,24 --export=ALL,SMOKE=1 --time=02:00:00 code/scripts/launch/hpc/varnet_loupe_lr_schedule_r8/run.sbatch
#   sbatch --array=0-32 code/scripts/launch/hpc/varnet_loupe_lr_schedule_r8/run.sbatch
#
#SBATCH --job-name=vlls_r8
#SBATCH --partition=gpu
#SBATCH --account=hpc_core
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96GB
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_log/%x-%A_%a.log

set -eo pipefail

: "${SLURM_ARRAY_TASK_ID:?submit with --array}"
EXP_DIR="${EXP_DIR:-$HOME/scratch/varnet_loupe_lr_schedule_r8}"
SEEDS=(42 43 44)
ARMS=(lr1e3_step lr5e4_step lr2e3_step lr5e4_wcos lr1e3_wcos lr2e3_wcos lr5e4_wsd lr1e3_wsd lr2e3_wsd lr1e3_const shared_2e3_step)
ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]:?task $SLURM_ARRAY_TASK_ID has no arm (valid: 0-$((3 * ${#ARMS[@]} - 1)))}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"

EPOCHS=400
case "$ARM" in
    lr5e4_*) LR=0.0005 ;;
    lr1e3_*) LR=0.001 ;;
    lr2e3_* | shared_2e3_*) LR=0.002 ;;
esac
MASK_LR=0.001
[[ "$ARM" == shared_* ]] && MASK_LR="$LR"
case "$ARM" in
    *_step)  SCHEDULE=(training.lr_scheduler=step training.lr_step_size=360 training.lr_gamma=0.1) ;;
    *_wcos)  SCHEDULE=(training.lr_scheduler=warmup_cosine training.lr_warmup_epochs=10) ;;
    *_wsd)   SCHEDULE=(training.lr_scheduler=wsd training.lr_warmup_epochs=10 training.lr_decay_fraction=0.2) ;;
    *_const) SCHEDULE=(training.lr_scheduler=constant) ;;
esac

NAME="${ARM}_s${SEED}"
OUT_DIR="$EXP_DIR/runs/$NAME"
VAL_EVERY=10
if [[ "${SMOKE:-0}" == 1 ]]; then
    OUT_DIR="$EXP_DIR/runs/smoke/$NAME-$SLURM_ARRAY_JOB_ID"
    EPOCHS=3
    VAL_EVERY=3
fi
if [[ -e "$OUT_DIR/last_model.pt" && "${SMOKE:-0}" != 1 ]]; then
    mv "$OUT_DIR" "$OUT_DIR.attempt-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUT_DIR"

export WANDB_MODE=offline
export WANDB_RUN_GROUP=varnet_loupe_lr_schedule_r8
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /mnt/software/ohpc/pub/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate dl

echo "arm=$ARM seed=$SEED lr=$LR mask_lr=$MASK_LR task=$SLURM_ARRAY_TASK_ID host=$(hostname) out=$OUT_DIR smoke=${SMOKE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$EXP_DIR/code"
python train.py --model sequential recon=varnet \
    datasets=fastmri_local_v1_flair_knee \
    "datasets.roots.hpc=\"$HOME/scratch/data/fastmri/{anatomy}/data\"" \
    "datasets.roots.local_batch2=$HOME/scratch/data/fastmri/knee/data" \
    datasets.limit_volumes=0.2 \
    'misc.resolution=[320,320]' \
    'sampler.accelerations=[8.0]' sampler.budget_over=acquired \
    sampler.sampler_type=loupe_2d sampler.loupe_vd_init_power=0 sampler.anatomy_conditioning=false \
    sampler.lr="$MASK_LR" \
    training.num_step=1 training.batch_size=16 training.num_epochs="$EPOCHS" \
    training.lr="$LR" "${SCHEDULE[@]}" \
    training.val_every="$VAL_EVERY" training.seed="$SEED" \
    "output.out_dir=$OUT_DIR" "output.run_name=vlls_$NAME"
