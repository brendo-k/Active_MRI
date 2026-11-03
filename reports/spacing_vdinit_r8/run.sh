#!/usr/bin/env bash
# Regenerates summary.md / summary.json from the nine training logs. The training itself is the SLURM array in
# ~/scratch/code/loupe_spacing_vdinit_r8 (job 2348509); delete summary.* to force a recompute. CPU only, a few seconds.
set -euo pipefail
EXP=${EXP:-$HOME/scratch/code/loupe_spacing_vdinit_r8}
cd "$(dirname "$0")"
python -I "$EXP/code/scripts/eval/summarize_spacing_vdinit.py" --runs "$EXP/runs" --out summary.json > summary.md
