#!/usr/bin/env bash
# Regenerates metrics.json and table.md from the three offline wandb runs (CPU only, seconds). The runs themselves are
# trained by the three sbatch commands below (one GPU, about 1.6 h each); skip those if the checkpoint directories exist.
# Delete reports/sense_mask_varnet/{metrics.json,table.md} to force a recompute.
set -euo pipefail
cd "$(dirname "$0")/../.."
LOGS=/home/remote/bkadota/scratch/logs/active_mri
# Training (from ~/jobs/active_mri; the launcher is reports/sense_mask_varnet/launch.sh):
#   for arm in pre_network null post_network; do
#       sbatch --job-name=sense_mask_$arm --export=ALL,ARM=$arm launch.sh; done
python scripts/eval/wandb_final_val_metrics.py --control pre_network --out reports/sense_mask_varnet \
    --run pre_network=$LOGS/sense_mask_varnet_pre_network_seed42 \
    --run null=$LOGS/sense_mask_varnet_null_seed42 \
    --run post_network=$LOGS/sense_mask_varnet_post_network_seed42
