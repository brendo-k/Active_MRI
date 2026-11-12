#!/usr/bin/env bash
# Long-schedule queue for reports/brain_static_vs_loupe (longschedule_proposal.md). It appends 6 jobs to the existing
# jobs.txt and restarts the two workers on GPUs 0-1 (earlier claims are kept, so no earlier run is rerun).
#   long_static_s<seed>   static's job line at 130 epochs, lr step at 116
#   long_uniform_s<seed>  loupe_uniform's job line at 130 epochs, lr step at 116
#
#   bash scripts/launch/queue_brain_static_vs_loupe_longschedule.sh <code snapshot dir>
set -euo pipefail
CODE="${1:?code snapshot dir}"
Q=/home/bkadota/data/seqmri_checkpoints/brain_static_vs_loupe
S=/home/bkadota/data/seqmri_checkpoints/loss_domain/runs_v2/loss_domain_rss_r8_flair_knee_100ep_seed42/best_model.pt
DEADLINE="${DEADLINE:-none}"
MINUTES=90
BASE="recon=varnet datasets=fastmri_local_v1_flair_knee misc.resolution=[320,320] sampler.accelerations=[8.0]
sampler.budget_over=acquired training.num_step=1 training.batch_size=16 sampler.sampler_type=loupe_2d
sampler.mask_train_mode=straight_through sampler.anatomy_conditioning=true datasets.train_anatomies=[brain]
datasets.limit_volumes=0.25 training.val_every=5 training.num_epochs=130 training.lr_step_size=116"
BASE=$(echo $BASE)
STATIC_MASK="sampler.fixed_mask_checkpoints={brain:$S,knee-PD:$S,knee-PDFS:$S}"
UNIFORM_START="sampler.loupe_vd_init_power=0"

if /usr/bin/grep -q '^long_static_s42|' "$Q/jobs.txt"; then
    echo "long-schedule jobs are already in $Q/jobs.txt" >&2
    exit 1
fi
{
    echo "# long schedule (longschedule_proposal.md)"
    for seed in 42 43 44; do
        echo "long_static_s$seed|$CODE|$MINUTES|$BASE $STATIC_MASK training.seed=$seed"
        echo "long_uniform_s$seed|$CODE|$MINUTES|$BASE $UNIFORM_START training.seed=$seed"
    done
} >> "$Q/jobs.txt"

rm -f "$Q/STOP"
for gpu in 0 1; do
    QUEUE_DIR="$Q" GPU="$gpu" DEADLINE="$DEADLINE" RUN_GROUP=brain_static_vs_loupe RUN_PREFIX=bsl_ \
        nohup setsid bash "$Q/bin/worker.sh" > "$Q/worker$gpu.longschedule.log" 2>&1 &
    sleep 90
done
echo "workers started; jobs in $Q/jobs.txt"
