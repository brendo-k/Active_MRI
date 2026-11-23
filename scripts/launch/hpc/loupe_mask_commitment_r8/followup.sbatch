#!/bin/bash
# LOUPE mask commitment follow-up (reports/loupe_mask_commitment_r8/followup_proposal.md): does `mix` gain once it trains
# longer, or once its flat-mask batches stop? 12-cascade lr 1e-3 step control as in run.sbatch, seeds 42/43/44.
#   tasks 0-2  control, 150 epochs (step at 135)      -> runs_e150/control_s<seed>
#   tasks 3-5  mix (flat fraction 0.25), 150 epochs   -> runs_e150/mix_s<seed>
#   tasks 6-8  mixdecay (0.25 -> 0 over epochs 0-30), 100 epochs (step at 90) -> runs_e100/mixdecay_s<seed>
#   task  9    control s44, 100 epochs                -> runs_e100/control_s44 (s42/s43 are round 1's runs/control_s4x)
#
#   cd ~/scratch/loupe_mask_commitment_r8
#   sbatch --array=0-9 --export=ALL,SMOKE=1 --time=00:40:00 code/scripts/launch/hpc/loupe_mask_commitment_r8/followup.sbatch
#   sbatch --array=0-9 code/scripts/launch/hpc/loupe_mask_commitment_r8/followup.sbatch
#
#SBATCH --job-name=lmc_r8_fu
#SBATCH --partition=gpu
#SBATCH --account=hpc_core
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96GB
#SBATCH --time=09:00:00
#SBATCH --output=slurm_log/%x-%A_%a.log

set -eo pipefail

: "${SLURM_ARRAY_TASK_ID:?submit with --array}"
EXP_DIR="${EXP_DIR:-$HOME/scratch/loupe_mask_commitment_r8}"
SEEDS=(42 43 44)
TASK="$SLURM_ARRAY_TASK_ID"

if (( TASK <= 8 )); then
    GROUPS_=(control_e150 mix_e150 mixdecay)
    GROUP="${GROUPS_[$((TASK / 3))]}"
    SEED="${SEEDS[$((TASK % 3))]}"
elif (( TASK == 9 )); then
    GROUP=control_e100
    SEED=44
else
    echo "task $TASK has no run (valid: 0-9)" >&2
    exit 1
fi

case "$GROUP" in
    control_e150) ARM=control; EPOCHS=150; STEP=135; RUNS_DIR=runs_e150; ARM_OVERRIDES=() ;;
    mix_e150) ARM=mix; EPOCHS=150; STEP=135; RUNS_DIR=runs_e150; ARM_OVERRIDES=(sampler.flat_mask_fraction=0.25) ;;
    mixdecay) ARM=mixdecay; EPOCHS=100; STEP=90; RUNS_DIR=runs_e100; ARM_OVERRIDES=(sampler.flat_mask_fraction=0.25) ;;
    control_e100) ARM=control; EPOCHS=100; STEP=90; RUNS_DIR=runs_e100; ARM_OVERRIDES=() ;;
esac

VAL_EVERY=5
DECAY_EPOCHS=30
if [[ -n "$SMOKE" ]]; then
    EPOCHS=2
    VAL_EVERY=1
    RUNS_DIR="smoke_$RUNS_DIR"
    ARM_OVERRIDES+=(training.limit_batches=0.02 datasets.val_limit_volumes=0.05)
    DECAY_EPOCHS=1
fi
if [[ "$GROUP" == mixdecay ]]; then
    ARM_OVERRIDES+=("sampler.flat_mask_decay_epochs=$DECAY_EPOCHS")
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

echo "group=$GROUP arm=$ARM seed=$SEED epochs=$EPOCHS step=$STEP task=$TASK host=$(hostname) out=$OUT_DIR overrides=${ARM_OVERRIDES[*]}"
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
    training.lr=0.001 training.lr_scheduler=step training.lr_step_size="$STEP" training.lr_gamma=0.1 \
    training.val_every="$VAL_EVERY" training.seed="$SEED" \
    "${ARM_OVERRIDES[@]}" \
    "output.out_dir=$OUT_DIR" "output.run_name=lmc_fu_${GROUP}_s$SEED"
