#!/bin/bash
# CoreHPC share of the loss-domain runs (reports/loss_domain/proposal.md): seeds 43 and 44 of every arm, plus seed 42
# of kspace_sense. The other seed-42 arms train locally (scripts/launch/queue_loss_domain_local.sh). One array task per
# (arm, seed); every task runs scripts/launch/train_loss_domain_r8_local_v1.sh with the CoreHPC data roots.
#
# Folder ~/scratch/loss_domain: code/ (snapshot of the branch, code/SNAPSHOT.txt), cache/ (the ESPIRiT image cache),
# runs/, slurm_log/. Submit from it:
#   cd ~/scratch/loss_domain
#   sbatch --array=10 --export=ALL,DEBUG=1 --time=02:00:00 code/scripts/launch/hpc/loss_domain/run_loss_domain.sbatch
#   sbatch --array=0-10%6 code/scripts/launch/hpc/loss_domain/run_loss_domain.sbatch
# DEBUG=1 trains 2% of one epoch into runs/debug/ (it also writes the loader's per-volume slice-energy caches, so
# the full tasks don't each read every volume at start-up).
#
#SBATCH --job-name=loss_domain
#SBATCH --partition=gpu
#SBATCH --account=hpc_core
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_log/%x-%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brenden.kadota@ucsf.edu

set -o pipefail

: "${SLURM_ARRAY_TASK_ID:?submit with --array, e.g. sbatch --array=0-10%6 run_loss_domain.sbatch}"
EXP_DIR="${EXP_DIR:-$HOME/scratch/loss_domain}"
TASKS=(rss:43 complex:43 kspace_sense:43 kspace_rss:43 kspace_rss_M:43
       rss:44 complex:44 kspace_sense:44 kspace_rss:44 kspace_rss_M:44
       kspace_sense:42)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]:?array index $SLURM_ARRAY_TASK_ID has no task (valid: 0-$((${#TASKS[@]} - 1)))}"
ARM="${TASK%%:*}"
SEED="${TASK##*:}"
EPOCHS=100
OUT_ROOT="$EXP_DIR/runs"
EXTRA=()
if [[ "${DEBUG:-0}" == 1 ]]; then
    OUT_ROOT="$EXP_DIR/runs/debug"
    EPOCHS=1
    EXTRA=(training.limit_batches=0.02)
fi
mkdir -p "$OUT_ROOT"

export WANDB_MODE=offline
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /mnt/software/ohpc/pub/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate dl

echo "task=$SLURM_ARRAY_TASK_ID arm=$ARM seed=$SEED host=$(hostname) out=$OUT_ROOT debug=${DEBUG:-0} code=$(cat "$EXP_DIR/code/SNAPSHOT.txt")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import sys, torch, hydra, fastmri; print('python', sys.version.split()[0], 'torch', torch.__version__, 'hydra', hydra.__version__, 'cuda', torch.cuda.is_available())"

ROOTS=('datasets.roots.hpc="~/scratch/data/fastmri/{anatomy}/data"' 'datasets.roots.local_batch2=~/scratch/data/fastmri/knee/data')
LAST_EPOCH="Epoch = \[ *$EPOCHS/ *$EPOCHS\]"
OUT="$OUT_ROOT/loss_domain_${ARM}_r8_flair_knee_${EPOCHS}ep_seed${SEED}"
cd "$EXP_DIR/code"
for attempt in 1 2; do
    FOREGROUND=1 GPU="${CUDA_VISIBLE_DEVICES:-0}" ARM="$ARM" SEED="$SEED" EPOCHS="$EPOCHS" OUT_ROOT="$OUT_ROOT" \
        COMPLEX_CACHE="$EXP_DIR/cache" scripts/launch/train_loss_domain_r8_local_v1.sh "${ROOTS[@]}" "${EXTRA[@]}"
    rc=$?
    if grep -qE "$LAST_EPOCH" "$OUT.log"; then
        echo "done $ARM seed $SEED (exit $rc)"
        exit 0
    fi
    mv "$OUT.log" "$OUT.attempt$attempt.log"
    tail -n 30 "$OUT.attempt$attempt.log"
    if ! grep -q 'out of memory' "$OUT.attempt$attempt.log"; then
        echo "$ARM seed $SEED stopped early (exit $rc); not retried in this job"
        exit 1
    fi
    echo "$ARM seed $SEED ran out of GPU memory; rerunning with recon.checkpoint_cascades=true"
    EXTRA+=(recon.checkpoint_cascades=true)
done
exit 1
