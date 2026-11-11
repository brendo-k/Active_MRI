#!/usr/bin/env bash
# Follow-up queue for reports/brain_static_vs_loupe (followup_proposal.md). It appends 4 jobs to the existing jobs.txt
# and restarts the two workers on GPUs 0-1 (the first round's claims are kept, so its 9 runs are not rerun).
#   frozen_learned_s<seed>  static's job line with the brain slot frozen to loupe_shared_s<seed>'s learned mask
#   static_repeat_s42       static_s42's job line again, at this code
# The workers exit at DEADLINE once no job is left.
#
#   bash scripts/launch/queue_brain_static_vs_loupe_followup.sh <code snapshot dir>
set -euo pipefail
CODE="${1:?code snapshot dir}"
Q=/home/bkadota/data/seqmri_checkpoints/brain_static_vs_loupe
S=/home/bkadota/data/seqmri_checkpoints/loss_domain/runs_v2/loss_domain_rss_r8_flair_knee_100ep_seed42/best_model.pt
DEADLINE="${DEADLINE:-2026-11-11 04:00}"
MINUTES=40
BASE="recon=varnet datasets=fastmri_local_v1_flair_knee misc.resolution=[320,320] sampler.accelerations=[8.0]
sampler.budget_over=acquired training.num_step=1 training.batch_size=16 sampler.sampler_type=loupe_2d
sampler.mask_train_mode=straight_through sampler.anatomy_conditioning=true datasets.train_anatomies=[brain]
datasets.limit_volumes=0.25 training.val_every=5 training.num_epochs=65 training.lr_step_size=58"
BASE=$(echo $BASE)
frozen() { echo "sampler.fixed_mask_checkpoints={brain:$1,knee-PD:$S,knee-PDFS:$S}"; }

if /usr/bin/grep -q '^frozen_learned_s42|' "$Q/jobs.txt"; then
    echo "follow-up jobs are already in $Q/jobs.txt" >&2
    exit 1
fi
{
    echo "# follow-up (followup_proposal.md)"
    for seed in 42 43 44; do
        echo "frozen_learned_s$seed|$CODE|$MINUTES|$BASE $(frozen "$Q/loupe_shared_s$seed/last_model.pt") training.seed=$seed"
    done
    echo "static_repeat_s42|$CODE|$MINUTES|$BASE $(frozen "$S") training.seed=42"
} >> "$Q/jobs.txt"

rm -f "$Q/STOP"
for gpu in 0 1; do
    QUEUE_DIR="$Q" GPU="$gpu" DEADLINE="$DEADLINE" RUN_GROUP=brain_static_vs_loupe RUN_PREFIX=bsl_ \
        nohup setsid bash "$Q/bin/worker.sh" > "$Q/worker$gpu.followup.log" 2>&1 &
    sleep 90
done
echo "workers started; jobs in $Q/jobs.txt"
