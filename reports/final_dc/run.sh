#!/usr/bin/env bash
# Regenerates every result file in reports/final_dc/. Needs GPUs 0-1 (this study's limit), the 10 checkpoints linked from
# checkpoints/ (~/data/seqmri_checkpoints/muon_layernorm_100ep and .../loss_domain/runs_v2), the local v1 raw data, and the
# two code trees the checkpoints trained with: the detached snapshot SeqMRI-worktrees/run-ml100-91932da and the
# exp/loss_domain worktree. Scoring skips existing reports/final_dc/scores/*.npz: delete them to rescore (about 1 h).
set -euo pipefail
cd "$(dirname "$0")/../.."

bash reports/final_dc/score.sh
source .venv/bin/activate
python scripts/eval/final_dc_summary.py reports/final_dc
