#!/usr/bin/env bash
# Regenerates every result file in reports/muon_layernorm_100ep/ from the finished runs in
# ~/data/seqmri_checkpoints/muon_layernorm_100ep (untracked; names, overrides and wandb ids in manifest.json).
# Needs one idle GPU (eval; this study stays on GPUs 0-1) and wandb access (curves). The eval reuses
# eval.per_slice.npz: delete it to rescore every checkpoint. The training runs themselves came from the queue in
# scripts/launch/varnet_train_speed_worker.sh (job lines: ~/data/seqmri_checkpoints/muon_layernorm_100ep/jobs.txt).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=reports/muon_layernorm_100ep
QUEUE=~/data/seqmri_checkpoints/muon_layernorm_100ep

# Final per-condition scores and paired differences, then validation curves and epoch times (wandb + run logs)
CUDA_VISIBLE_DEVICES=0 python scripts/eval/varnet_train_speed_eval.py --queue-dir $QUEUE --out $OUT/eval.json
python scripts/eval/varnet_train_speed_curves.py --queue-dir $QUEUE --prefix ml100_ --out $OUT/curves.json

# Frozen criteria, tables and figures
python scripts/eval/muon_layernorm_100ep_summary.py $OUT --queue-dir $QUEUE
mkdir -p $OUT/figures
python scripts/plots/varnet_train_speed_curves.py $OUT/curves.json $OUT/figures/curves.png \
    control,muon,layer,muon_layer 'all of local v1 FLAIR+knee (val 1,938 slices)'
python scripts/plots/muon_layernorm_100ep_lead.py $OUT/curves.json $OUT/figures/lead_over_control.png
