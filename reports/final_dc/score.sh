#!/usr/bin/env bash
# Scores every checkpoint of reports/final_dc/proposal.md with its own training code, on GPUs 0-1 (this study's limit).
# GPU 0: the 2x2's control and muon arms, then loss_domain rss. GPU 1: layer and muon_layer, then loss_domain kspace_gt.
# Existing OUT/*.npz are skipped: delete them to rescore.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$HERE/../../scripts/eval/final_dc_eval.py"
OUT="$HERE/scores"
LOGS="$HERE/logs"
ML100_CODE=/home/bkadota/code/active_mri/SeqMRI-worktrees/run-ml100-91932da
LOSS_DOMAIN_CODE=/home/bkadota/code/active_mri/SeqMRI-worktrees/exp-loss_domain
ML100=~/data/seqmri_checkpoints/muon_layernorm_100ep
LOSS_DOMAIN=~/data/seqmri_checkpoints/loss_domain/runs_v2
mkdir -p "$OUT" "$LOGS"
source /home/bkadota/code/active_mri/SeqMRI/.venv/bin/activate

chain() {
    local gpu=$1 arm_a=$2 arm_b=$3 loss_arm=$4
    (cd "$ML100_CODE" && CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" --code-root . --loader ml100 --tag ml100 --out "$OUT" \
        --runs $ML100/${arm_a}_s42 $ML100/${arm_a}_s43 $ML100/${arm_b}_s42 $ML100/${arm_b}_s43)
    (cd "$LOSS_DOMAIN_CODE" && CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" --code-root . --loader loss_domain --tag loss_domain \
        --out "$OUT" --runs $LOSS_DOMAIN/loss_domain_${loss_arm}_r8_flair_knee_100ep_seed42)
}

chain 0 control muon rss > "$LOGS/gpu0.log" 2>&1 &
chain 1 layer muon_layer kspace_gt > "$LOGS/gpu1.log" 2>&1 &
wait
