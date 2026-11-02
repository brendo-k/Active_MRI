#!/bin/bash
# Usage: sbatch --job-name=sense_mask_$ARM --export=ALL,ARM=<pre_network|null|post_network> sense_mask_varnet.sh
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=./slurm_log/%x-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brenden.kadota@ucsf.edu

WORKDIR="/mnt/scratch/user/bkadota/code/active_MRI_worktrees/exp-sense_mask_varnet"
DATA="/home/remote/bkadota/scratch/data/fastmri"
OUT_DIR="/home/remote/bkadota/scratch/logs/active_mri/sense_mask_varnet_${ARM}_seed42"
mkdir -p "$OUT_DIR"
RESUME_ARGS=()
[ -f "$OUT_DIR/last_model.pt" ] && RESUME_ARGS=(output.checkpoint="$OUT_DIR/last_model.pt") && echo "Resuming from $OUT_DIR/last_model.pt"
export WANDB_MODE=offline

source /mnt/software/ohpc/pub/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate dl
cd "$WORKDIR"
python train.py --model sequential \
    recon=varnet recon.num_cascades=12 recon.csme=false \
    misc.resolution=[320,320] \
    datasets.limit_volumes=0.5 \
    "datasets.roots.hpc=${DATA}/{anatomy}/data" datasets.roots.local_batch2=${DATA}/knee/data \
    training.num_step=1 training.batch_size=16 training.num_epochs=50 \
    training.lr=0.001 training.lr_scheduler=step training.lr_step_size=40 training.seed=42 \
    loss.loss_type=l1 \
    sampler.sampler_type=loupe_2d sampler.mask_train_mode=straight_through sampler.loupe_vd_init_power=0 \
    sampler.accelerations=[8.0] sampler.budget_over=acquired sampler.anatomy_conditioning=false sampler.coil_combine=sense \
    sampler.sense_mask_method=${ARM} \
    output.out_dir="$OUT_DIR" output.run_name="sense_mask_${ARM}_seed42" \
    "${RESUME_ARGS[@]}"
