#!/bin/bash
# Tier-L extension of brain_varnet_width (reports/brain_varnet_width/proposal_tier_l.md): brain-only VarNets on the
# frozen shared mask S, {6, 12} cascades x {18, 26} channels on the 300 tier-L FLAIR volumes, seeds 42 and 43, plus
# 6 x 18 on tier S as the CoreHPC platform control. 100 epochs, lr x0.1 at epoch 90, val every 5 epochs on the pinned
# 10-volume brain subset, early-stop patience 5. One array task per run:
#   0-7  tier L: task = 2 * cell + seed index, cells 6x18 6x26 12x18 12x26
#   8-9  tier S 6x18, seeds 42 and 43
#
# Layout under $EXP_DIR: code/ (a clone of exp/brain_varnet_width at the run-state commit), mask_S/best_model.pt,
# runs_L/<name>, runs_S/<name>, slurm_log/.
#
#   cd ~/scratch/brain_varnet_width_l
#   sbatch --array=0 --export=ALL,DEBUG=1 --time=00:30:00 code/scripts/launch/hpc/brain_varnet_width_l/run.sbatch
#   sbatch --array=0-9 code/scripts/launch/hpc/brain_varnet_width_l/run.sbatch
#
#SBATCH --job-name=bvw_l
#SBATCH --partition=gpu
#SBATCH --account=hpc_core
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96GB
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_log/%x-%A_%a.log

set -eo pipefail

: "${SLURM_ARRAY_TASK_ID:?submit with --array}"
EXP_DIR="${EXP_DIR:-$HOME/scratch/brain_varnet_width_l}"
MASK="$EXP_DIR/mask_S/best_model.pt"
[[ -e "$MASK" ]] || { echo "$MASK not found" >&2; exit 1; }

CELLS=(6:18 6:26 12:18 12:26)
SEEDS=(42 43)
if (( SLURM_ARRAY_TASK_ID < 8 )); then
    TIER=L
    CELL="${CELLS[$((SLURM_ARRAY_TASK_ID / 2))]}"
    SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}"
elif (( SLURM_ARRAY_TASK_ID < 10 )); then
    TIER=S
    CELL=6:18
    SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID - 8))]}"
else
    echo "task $SLURM_ARRAY_TASK_ID has no run (valid: 0-9)" >&2; exit 1
fi
CASCADES=${CELL%:*}
CHANS=${CELL#*:}
NAME="c${CASCADES}_ch${CHANS}_s${SEED}"
OUT_DIR="$EXP_DIR/runs_$TIER/$NAME"
EXTRA=()
if [[ "${DEBUG:-0}" == 1 ]]; then
    OUT_DIR="$EXP_DIR/debug/runs_$TIER/$NAME-$SLURM_ARRAY_JOB_ID"
    EXTRA=(training.num_epochs=1 training.limit_batches=0.02)
elif [[ -e "$OUT_DIR/last_model.pt" ]]; then
    mv "$OUT_DIR" "$OUT_DIR.attempt-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUT_DIR"

export WANDB_MODE=offline
export WANDB_RUN_GROUP=brain_varnet_width_l
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /mnt/software/ohpc/pub/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate dl

echo "tier=$TIER cell=$CELL seed=$SEED task=$SLURM_ARRAY_TASK_ID host=$(hostname) out=$OUT_DIR debug=${DEBUG:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
sha256sum "$MASK"

cd "$EXP_DIR/code"
git log --oneline -1
python train.py --model sequential recon=varnet \
    datasets=fastmri_local_v1_flair_knee \
    "datasets.roots.hpc=\"$HOME/scratch/data/fastmri/{anatomy}/data\"" \
    "datasets.roots.local_batch2=$HOME/scratch/data/fastmri/knee/data" \
    datasets.train_tier="$TIER" datasets.limit_volumes=1.0 \
    'datasets.train_anatomies=[brain]' 'datasets.val_anatomies=[brain]' datasets.val_limit_volumes=0.25 \
    'misc.resolution=[320,320]' \
    'sampler.accelerations=[8.0]' sampler.budget_over=acquired \
    sampler.sampler_type=loupe_2d sampler.mask_train_mode=straight_through sampler.anatomy_conditioning=true \
    "sampler.fixed_mask_checkpoints={brain:$MASK,knee-PD:$MASK,knee-PDFS:$MASK}" \
    training.num_step=1 training.batch_size=16 training.num_epochs=100 training.lr_step_size=90 \
    training.val_every=5 training.early_stop_patience=5 training.seed="$SEED" \
    recon.num_cascades="$CASCADES" recon.chans="$CHANS" \
    "output.out_dir=$OUT_DIR" "output.run_name=bvwl_${TIER}_$NAME" \
    "${EXTRA[@]}"
