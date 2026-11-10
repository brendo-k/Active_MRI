#!/usr/bin/env bash
# Queue for reports/brain_static_vs_loupe: 3 arms x 3 seeds of a brain-only VarNet on a quarter of local v1, R=8.
#   static         the shared mask S, frozen (sampled from its probabilities every step)
#   loupe_shared   LOUPE started from S
#   loupe_uniform  LOUPE started from uniform noise (main's default start)
# Smoke test (2026-11-10, 2 epochs, 2 GPUs): ~36 s train + ~22 s val per epoch after a ~2 min compile, so val_every=5
# and 65 epochs give ~46 min per run, and 9 runs fit ~4 h on GPUs 0-1. Seed-major order keeps whole seed triplets.
#
#   bash scripts/launch/queue_brain_static_vs_loupe.sh <code snapshot dir>
set -euo pipefail
CODE="${1:?code snapshot dir}"
Q=/home/bkadota/data/seqmri_checkpoints/brain_static_vs_loupe
S=/home/bkadota/data/seqmri_checkpoints/loss_domain/runs_v2/loss_domain_rss_r8_flair_knee_100ep_seed42/best_model.pt
EPOCHS=65
LR_STEP=58
MINUTES=50
BASE="recon=varnet datasets=fastmri_local_v1_flair_knee misc.resolution=[320,320] sampler.accelerations=[8.0]
sampler.budget_over=acquired training.num_step=1 training.batch_size=16 sampler.sampler_type=loupe_2d
sampler.mask_train_mode=straight_through sampler.anatomy_conditioning=true datasets.train_anatomies=[brain]
datasets.limit_volumes=0.25 training.val_every=5 training.num_epochs=$EPOCHS training.lr_step_size=$LR_STEP"
BASE=$(echo $BASE)
declare -A ARM=(
    [static]="sampler.fixed_mask_checkpoints={brain:$S,knee-PD:$S,knee-PDFS:$S}"
    [loupe_shared]="sampler.init_mask_checkpoint=$S"
    [loupe_uniform]="sampler.loupe_vd_init_power=0"
)

mkdir -p "$Q/bin" "$Q/claims"
{
    echo "# name|code snapshot|estimated minutes|overrides. GPUs 0-1, seed-major."
    for seed in 42 43 44; do
        for arm in static loupe_shared loupe_uniform; do
            echo "${arm}_s$seed|$CODE|$MINUTES|$BASE ${ARM[$arm]} training.seed=$seed"
        done
    done
} > "$Q/jobs.txt"

for gpu in 0 1; do
    QUEUE_DIR="$Q" GPU="$gpu" DEADLINE=none RUN_GROUP=brain_static_vs_loupe RUN_PREFIX=bsl_ \
        nohup setsid bash "$Q/bin/worker.sh" > "$Q/worker$gpu.log" 2>&1 &
    sleep 90
done
echo "workers started; jobs in $Q/jobs.txt"
