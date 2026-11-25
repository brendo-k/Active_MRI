#!/bin/bash
# Regenerates every result file in this folder. Stages that need checkpoints or a GPU run only with WITH_CHECKPOINTS=1.
#
# scores/*.{json,npz} and figures/scorer/ are the scorer's output: each run was scored on CoreHPC by train.sbatch right
# after training, on the same card, at run state 2 (24db7c7):
#   python scripts/eval/eval_multistep_adaptivity.py score --run <run dir> --name <run> --out <scores> --gradients
# Rescoring needs the checkpoints (CoreHPC ~/scratch/loss_domain__loupe_loss_partial_fourier/runs, lab-server copy
# /export/scratch/bkadota/loss_domain__loupe_loss_partial_fourier_sync/runs) and a GPU.
# probe_fscr1.json was written at run state 1 (see addendum_2026-11-20_probe.md).
# Delete contrasts.json, summary.*, curves.json, analysis/layout.json, analysis/probe_L*.json, analysis/extent.json, analysis/phase_realness.json and figures/*.png to force
# a recompute; every command below overwrites its outputs anyway.
set -euo pipefail
cd "$(dirname "$0")/../.."

RUNS_DIR=${RUNS_DIR:-/export/scratch/bkadota/loss_domain__loupe_loss_partial_fourier_sync/runs}
DATA_ROOT=${DATA_ROOT:-'hpc=/export/scratch/bkadota/fastmri_raw/{anatomy}'}

python scripts/eval/eval_multistep_adaptivity.py contrast --scores reports/loss_domain__loupe_loss_partial_fourier/scores --out reports/loss_domain__loupe_loss_partial_fourier/contrasts.json --pair half_rss_s42=Lrss_1_s42_tF-Lrsssym_1_s42_tF --pair half_rss_s43=Lrss_1_s43_tF-Lrsssym_1_s43_tF --pair half_cplx_s42=Lcplx_1_s42_tF-Lcplxsym_1_s42_tF --pair half_cplx_s43=Lcplx_1_s43_tF-Lcplxsym_1_s43_tF

if [[ "${WITH_CHECKPOINTS:-0}" == 1 ]]; then
  python scripts/diagnostics/loupe_map_asymmetry.py ${RUNS_DIR}/Lrss_1_s42_tF ${RUNS_DIR}/Lrss_1_s43_tF ${RUNS_DIR}/Lcplx_1_s42_tF ${RUNS_DIR}/Lcplx_1_s43_tF ${RUNS_DIR}/Lrsssym_1_s42_tF ${RUNS_DIR}/Lrsssym_1_s43_tF ${RUNS_DIR}/Lcplxsym_1_s42_tF ${RUNS_DIR}/Lcplxsym_1_s43_tF --out reports/loss_domain__loupe_loss_partial_fourier/analysis/layout.json
  for arm in Lrss Lcplx Lrsssym Lcplxsym; do
    python scripts/diagnostics/conjugate_symmetry_probe.py --run ${RUNS_DIR}/${arm}_1_s42_tF --run ${RUNS_DIR}/${arm}_1_s43_tF --out reports/loss_domain__loupe_loss_partial_fourier/analysis/probe_${arm}.json
  done
  python reports/loss_domain__loupe_loss_partial_fourier/analysis/summarise.py maps extent recons --runs-dir ${RUNS_DIR} --data-root "${DATA_ROOT}"
  python reports/loss_domain__loupe_loss_partial_fourier/analysis/phase_realness.py --runs-dir ${RUNS_DIR} --data-root "${DATA_ROOT}" --out reports/loss_domain__loupe_loss_partial_fourier/analysis/phase_realness.json
fi
python reports/loss_domain__loupe_loss_partial_fourier/analysis/summarise.py tables curves
