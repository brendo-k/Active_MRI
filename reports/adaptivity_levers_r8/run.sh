#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# CPU only (about 15 minutes with 4 x 40 workers); reads the 120 local-v1 validation volumes from /export/scratch.
# Every stage overwrites its output, so nothing needs deleting to force a recompute.
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/diagnostics/encoding_census_r8.py support --workers 40 --out reports/adaptivity_levers_r8/support_census.json &
.venv/bin/python scripts/diagnostics/encoding_census_r8.py gfactor --acceleration 8 --workers 40 --out reports/adaptivity_levers_r8/gfactor_census_r8.json &
.venv/bin/python scripts/diagnostics/encoding_census_r8.py gfactor --acceleration 4 --workers 40 --out reports/adaptivity_levers_r8/gfactor_census_r4.json &
.venv/bin/python scripts/diagnostics/encoding_census_r8.py roi --workers 40 --out reports/adaptivity_levers_r8/roi_gfactor_r8.json &
wait
.venv/bin/python scripts/diagnostics/encoding_census_r8.py summarize --dir reports/adaptivity_levers_r8

# Snapshots of the multi-step evidence, pinned to the tip of exp/multistep_adaptivity (c718a3d). They are copies of
# results that branch produced on CoreHPC; nothing is recomputed here.
git show c718a3d:reports/multistep_adaptivity/kscr1/contrasts.txt > reports/adaptivity_levers_r8/multistep_kscr1_contrasts.txt
git show c718a3d:reports/multistep_adaptivity/kscr2/contrasts.txt > reports/adaptivity_levers_r8/multistep_kscr2_contrasts.txt
git show c718a3d:reports/multistep_adaptivity/kscr2/leak/contrasts_gap.txt > reports/adaptivity_levers_r8/multistep_kscr2_leak_gap.txt
git show c718a3d:reports/multistep_adaptivity/kscr2/leak/contrasts_swap.txt > reports/adaptivity_levers_r8/multistep_kscr2_leak_swap.txt
.venv/bin/python scripts/report_manifest.py check reports/adaptivity_levers_r8
