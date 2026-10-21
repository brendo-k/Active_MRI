#!/usr/bin/env bash
# Held-out per-volume parametric mask oracle (block split): the pattern families an ACS -> parameters predictor would
# output, fitted on F and scored on H, with other-subject controls and an H-only fit for parameter reliability.
#   bash analysis/subject_coadapt/run_param_family.sh <gpu> <anatomy> <start:stop>
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
gpu=$1; anatomy=$2; volumes=$3
log=analysis/subject_coadapt/param_family_${anatomy}_${volumes/:/-}.log
echo "start param_family $anatomy $volumes gpu$gpu $(date)" >> analysis/subject_coadapt/launcher.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/subject_coadapt_ceiling_r8.py \
  --anatomy $anatomy --volumes $volumes --split block \
  --arms base,mask,mask_in,param,param_in,param_ctrl,param_h,grid,grid_in,grid_ctrl,grid_h --tag param > $log 2>&1
rc=$?
echo "done param_family $anatomy $volumes gpu$gpu rc=$rc" >> analysis/subject_coadapt/launcher.log
