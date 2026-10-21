#!/bin/bash
# SDUM (NV-Raw2Insights-MRI small, fine-tuned from the released weights) as the reconstructor under a learned 2D LOUPE
# mask at R=8 on balanced brain T2 + knee: the like-for-like of loupe2d_varnet_r8_brain_knee[_anatcond] (same data,
# acceleration, sampler, sampler lr, epochs). One array task per arm:
#   0 shared       one LOUPE mask, SDUM sees acq_type T2w for every slice
#   1 anatcond     one LOUPE mask per anatomy, SDUM sees T2w for every slice
#   2 anatcond_uc  one LOUPE mask per anatomy, and SDUM's universal conditioning gets brain->T2w, knee->T1w
#
# Cost: about 7.4 s per slice for forward + backward on an L40S (22 GiB at batch 2, 43 GiB at batch 4), so one epoch of the
# 2700 training slices is about 5.5 h plus 20 min of validation there. 8 epochs at batch 2 gives about the same number of
# mask updates (~11k) as the 50-epoch batch-16 VarNet baselines, not the same number of slices seen. lr 1e-4 for the
# pretrained SDUM, 1e-3 for the mask (as the baselines). last_model.pt is written every epoch, so a timeout still
# leaves a usable checkpoint.
#
# Layout under $EXP_DIR: code/ (this repo snapshot), NV-Raw2insights-MRI/ (clone), ckpt/nv_raw2insights_mri_small.pt
#
#   cd ~/scratch/sdum_r8
#   sbatch --array=0 --export=ALL,DEBUG=1 --time=01:00:00 code/scripts/launch/hpc/sdum_r8/run_sdum_r8.sbatch  # smoke test
#   sbatch --array=0-2 code/scripts/launch/hpc/sdum_r8/run_sdum_r8.sbatch
#
#SBATCH --job-name=sdum_r8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm_log/%x-%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brenden.kadota@ucsf.edu

set -eo pipefail

: "${SLURM_ARRAY_TASK_ID:?submit with --array, e.g. sbatch --array=0-2 run_sdum_r8.sbatch}"
EXP_DIR="${EXP_DIR:-$HOME/scratch/sdum_r8}"
DATA_ROOT="${DATA_ROOT:-$HOME/scratch/loupe_spacing_vdinit_r8/data/fastmri_r8split}"
SDUM_REPO="${SDUM_REPO:-$EXP_DIR/NV-Raw2insights-MRI}"
SDUM_CKPT="${SDUM_CKPT:-$EXP_DIR/ckpt/nv_raw2insights_mri_small.pt}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-8}"
for path in "$DATA_ROOT" "$SDUM_REPO" "$SDUM_CKPT"; do
    [[ -e "$path" ]] || { echo "$path not found" >&2; exit 1; }
done

ARMS=(shared anatcond anatcond_uc)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]:?array index $SLURM_ARRAY_TASK_ID has no arm (valid: 0-$((${#ARMS[@]} - 1)))}"
case "$ARM" in
    shared)      ARM_OVERRIDES=(sampler.anatomy_conditioning=false recon.anatomy_conditioning=false) ;;
    anatcond)    ARM_OVERRIDES=(sampler.anatomy_conditioning=true recon.anatomy_conditioning=false) ;;
    anatcond_uc) ARM_OVERRIDES=(sampler.anatomy_conditioning=true recon.anatomy_conditioning=true) ;;
esac

OUT_DIR="$EXP_DIR/runs/${ARM}-${SLURM_ARRAY_JOB_ID}"
if [[ "${DEBUG:-0}" == 1 ]]; then
    OUT_DIR="$EXP_DIR/runs/debug/${ARM}-${SLURM_ARRAY_JOB_ID}"
    ARM_OVERRIDES+=("training.num_epochs=1" "training.limit_batches=0.02")
fi
mkdir -p "$OUT_DIR"

export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /mnt/software/ohpc/pub/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate dl

echo "arm=$ARM task=$SLURM_ARRAY_TASK_ID host=$(hostname) out=$OUT_DIR debug=${DEBUG:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$EXP_DIR/code"
python train.py --model loupe recon=sdum \
    "recon.repo_path=$SDUM_REPO" "recon.checkpoint=$SDUM_CKPT" \
    datasets=brain_knee_balanced \
    "datasets.data_root=$DATA_ROOT" \
    datasets.num_workers=16 \
    'sampler.accelerations=[8.0]' \
    sampler.sampler_type=loupe_2d \
    sampler.lr=0.001 \
    training.lr=0.0001 \
    training.lr_step_size=6 \
    training.batch_size="$BATCH_SIZE" \
    training.num_epochs="$EPOCHS" \
    "output.out_dir=$OUT_DIR" \
    "${ARM_OVERRIDES[@]}"
