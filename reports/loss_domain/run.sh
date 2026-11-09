#!/usr/bin/env bash
# Regenerates the round-2 results of reports/loss_domain (see report.md). Needs the four seed-42 checkpoints in
# ~/data/seqmri_checkpoints/loss_domain/runs_v2, the round-1 rss checkpoint in .../loss_domain/runs (mask_overlap.py
# model check), the v2 ESPIRiT cache /export/scratch/bkadota/fastmri_complex_espirit_v2, the local v1 raw data and one
# GPU. The eval caches each run's per-slice scores in eval_v2/runs/; delete that folder to force a recompute.
# Round-1 files (eval/, qc/, the other diag/ scripts and outputs) were produced at earlier commits and are not rerun.
#
# Training (about 12.7 h per run on an A6000): bash scripts/launch/queue_loss_domain_local.sh, which runs
# scripts/launch/train_loss_domain_r8_local_v1.sh for rss, rss_M, kspace_gt and complex at seed 42.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

python scripts/eval/eval_loss_domain.py --runs-root ~/data/seqmri_checkpoints/loss_domain/runs_v2 --out reports/loss_domain/eval_v2 > reports/loss_domain/eval_v2/eval.log 2>&1
python reports/loss_domain/diag/intensity_fit.py ~/data/seqmri_checkpoints/loss_domain/runs_v2 rss rss_M kspace_gt complex > reports/loss_domain/diag/intensity_fit_v2.txt 2>&1
python reports/loss_domain/diag/mask_overlap.py > reports/loss_domain/diag/mask_overlap.txt 2>&1
python reports/loss_domain/diag/plot_masks.py --out reports/loss_domain/diag/masks.png
python reports/loss_domain/diag/complex_ssim.py ~/data/seqmri_checkpoints/loss_domain/runs_v2 > reports/loss_domain/diag/complex_ssim_v2.txt 2>&1
python reports/loss_domain/diag/complex_cascades.py ~/data/seqmri_checkpoints/loss_domain/runs_v2 > reports/loss_domain/diag/complex_cascades_v2.txt 2>&1
