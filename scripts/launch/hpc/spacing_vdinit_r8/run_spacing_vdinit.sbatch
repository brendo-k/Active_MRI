#!/bin/bash
# Spacing test + VD-init LOUPE at R=8: VarNet on balanced brain T2 + knee, the recipe of the local reference run
# checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05 (50 epochs, lr 1e-3 step at 40, batch 16, slope 5,
# grad floors 0.05). One array task per arm; run.md has the design and how to read the results.
#
# Submit from the experiment folder (the log path is relative to it):
#   cd ~/scratch/loupe_spacing_vdinit_r8
#   sbatch --array=2,4 --export=ALL,DEBUG=1 --time=01:00:00 run_spacing_vdinit.sbatch   # smoke test, 1 short epoch
#   sbatch --array=0-8%4 run_spacing_vdinit.sbatch                                      # all arms, 4 at a time
#
#SBATCH --job-name=spacing_vdinit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm_log/%x-%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brenden.kadota@ucsf.edu

set -eo pipefail

: "${SLURM_ARRAY_TASK_ID:?submit with --array, e.g. sbatch --array=0-8%4 run_spacing_vdinit.sbatch}"
EXP_DIR="${EXP_DIR:-$HOME/scratch/loupe_spacing_vdinit_r8}"
# Default: symlinks reproducing the local data layout, so the balanced split is the local one (data/README.txt).
# A DATA_ROOT with more volumes builds a bigger balanced split (several times the slices per epoch).
DATA_ROOT="${DATA_ROOT:-$EXP_DIR/data/fastmri_r8split}"
[[ -d "$DATA_ROOT" ]] || { echo "DATA_ROOT $DATA_ROOT not found" >&2; exit 1; }

# Array index -> arm. random/vdinit: LOUPE trained jointly (seed in the name). The rest: VarNet trained from
# scratch on one fixed mask from masks/ (training seed 42, the mask seed is in the name).
ARMS=(random_s1 random_s2 vdinit_s1 vdinit_s2 pd_loupe_s0 pd_loupe_s1 white_loupe_s0 white_loupe_s1 topk_loupe)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]:?array index $SLURM_ARRAY_TASK_ID has no arm (valid: 0-$((${#ARMS[@]} - 1)))}"
case "$ARM" in
    random_s*) ARM_OVERRIDES=("training.seed=${ARM#random_s}") ;;
    vdinit_s*) ARM_OVERRIDES=("training.seed=${ARM#vdinit_s}" "sampler.loupe_vd_init_power=4") ;;
    *)         ARM_OVERRIDES=("training.seed=42" "sampler.fixed_mask_path=$EXP_DIR/masks/$ARM.pt") ;;
esac

OUT_DIR="$EXP_DIR/runs/${ARM}-${SLURM_ARRAY_JOB_ID}"
if [[ "${DEBUG:-0}" == 1 ]]; then
    OUT_DIR="$EXP_DIR/runs/debug/${ARM}-${SLURM_ARRAY_JOB_ID}"
    ARM_OVERRIDES+=("training.num_epochs=1" "training.limit_batches=0.02")
fi
mkdir -p "$OUT_DIR"

export WANDB_MODE=offline
export OMP_NUM_THREADS=1  # the 16 loader workers each run small FFTs; keep BLAS from oversubscribing the cores
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /mnt/software/ohpc/pub/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate dl

echo "arm=$ARM task=$SLURM_ARRAY_TASK_ID host=$(hostname) out=$OUT_DIR debug=${DEBUG:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import sys, torch, hydra; print('python', sys.version.split()[0], 'torch', torch.__version__, 'hydra', hydra.__version__, 'cuda', torch.cuda.is_available())"

cd "$EXP_DIR/code"
python train.py --model loupe --recon varnet --dataset brain_knee_balanced --acceleration 8 \
    --batch-size 16 --epochs 50 \
    "datasets.data_root=$DATA_ROOT" \
    datasets.num_workers=16 \
    sampler.slope=5 \
    sampler.mask_grad_floor=0.05 \
    sampler.ste_grad_floor=0.05 \
    "output.out_dir=$OUT_DIR" \
    "${ARM_OVERRIDES[@]}"
