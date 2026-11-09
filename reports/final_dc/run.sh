#!/usr/bin/env bash
# Regenerates every result file in reports/final_dc/. Needs GPUs 0-1 (this study's limit), the 10 checkpoints linked from
# checkpoints/ (~/data/seqmri_checkpoints/muon_layernorm_100ep and .../loss_domain/runs_v2), the local v1 raw data, and the
# two code trees the checkpoints trained with: the detached snapshot SeqMRI-worktrees/run-ml100-91932da and the
# exp/loss_domain worktree. Each checkpoint is scored from its own code tree as working directory, because both loaders
# resolve the manifest relative to it. Scoring skips existing reports/final_dc/scores/*.npz: delete them to rescore (about 1 h).
# GPU 0: the 2x2's control and muon arms, then loss_domain rss. GPU 1: layer and muon_layer, then loss_domain kspace_gt.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
OUT="$REPO/reports/final_dc/scores"
LOGS="$REPO/reports/final_dc/logs"
ML100_CODE=/home/bkadota/code/active_mri/SeqMRI-worktrees/run-ml100-91932da
LOSS_DOMAIN_CODE=/home/bkadota/code/active_mri/SeqMRI-worktrees/exp-loss_domain
ML100=~/data/seqmri_checkpoints/muon_layernorm_100ep
LOSS_DOMAIN=~/data/seqmri_checkpoints/loss_domain/runs_v2
mkdir -p "$OUT" "$LOGS"
source .venv/bin/activate

chain() {
    local gpu=$1 arm_a=$2 arm_b=$3 loss_arm=$4
    (cd "$ML100_CODE" && CUDA_VISIBLE_DEVICES=$gpu python "$REPO/scripts/eval/final_dc_eval.py" --code-root . --loader ml100 \
        --tag ml100 --out "$OUT" --runs $ML100/${arm_a}_s42 $ML100/${arm_a}_s43 $ML100/${arm_b}_s42 $ML100/${arm_b}_s43)
    (cd "$LOSS_DOMAIN_CODE" && CUDA_VISIBLE_DEVICES=$gpu python "$REPO/scripts/eval/final_dc_eval.py" --code-root . \
        --loader loss_domain --tag loss_domain --out "$OUT" --runs $LOSS_DOMAIN/loss_domain_${loss_arm}_r8_flair_knee_100ep_seed42)
}

chain 0 control muon rss > "$LOGS/gpu0.log" 2>&1 &
chain 1 layer muon_layer kspace_gt > "$LOGS/gpu1.log" 2>&1 &
wait

python scripts/eval/final_dc_summary.py reports/final_dc
