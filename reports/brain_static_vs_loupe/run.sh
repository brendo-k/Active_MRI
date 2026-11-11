#!/usr/bin/env bash
# Regenerates every result file in reports/brain_static_vs_loupe/ from the finished runs in
# ~/data/seqmri_checkpoints/brain_static_vs_loupe (untracked; names, overrides and wandb ids in manifest.json).
# The runs came from scripts/launch/queue_brain_static_vs_loupe.sh (job lines in that folder's jobs.txt).
# Needs one idle GPU for the eval (~5 min per run). Delete eval.json and eval_per_slice.pt to force a rescore.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=reports/brain_static_vs_loupe
QUEUE=~/data/seqmri_checkpoints/brain_static_vs_loupe

# last_model.pt of all 9 runs on the full brain val set (40 volumes), paired contrasts, budget relocated
CUDA_VISIBLE_DEVICES=0 python scripts/eval/brain_static_vs_loupe_eval.py --queue $QUEUE --seeds 42 43 44 --out $OUT
# Frozen criteria, seed spread, seed swap, paired t, S support check, mean-map figure
python scripts/eval/brain_static_vs_loupe_summary.py --report $OUT --queue $QUEUE --shared ~/data/seqmri_checkpoints/loss_domain/runs_v2/loss_domain_rss_r8_flair_knee_100ep_seed42/best_model.pt
