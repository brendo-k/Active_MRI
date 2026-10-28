#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# This planning report creates no new reconstruction results. These commands
# refresh copies of historical evidence, overwriting the three snapshots.
# No deletion is needed. Changed source files require evidence review and a
# regenerated manifest; the final check deliberately detects such changes.
rtk proxy cp reports/ismrm2027/anatomy_pair_seed42.json reports/future_work/anatomy_pair_seed42.json
rtk proxy cp reports/ismrm2027/anatomy_pair_seed43.json reports/future_work/anatomy_pair_seed43.json
rtk proxy cp reports/ismrm2027/vdinit_summary.json reports/future_work/vdinit_summary.json
rtk proxy .venv/bin/python scripts/report_manifest.py check reports/future_work
