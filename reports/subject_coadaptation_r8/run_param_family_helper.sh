#!/usr/bin/env bash
# Helper for run_param_family.sh: works through shards last volume first (--reverse), skipping finished volumes.
#   bash reports/subject_coadaptation_r8/run_param_family_helper.sh <gpu> <anatomy>:<start:stop> [<anatomy>:<start:stop> ...]
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
gpu=$1; shift
for shard in "$@"; do
  anatomy=${shard%%:*}; volumes=${shard#*:}
  log=reports/subject_coadaptation_r8/param_family_helper_${anatomy}_${volumes/:/-}.log
  echo "start param_family helper $anatomy $volumes gpu$gpu" >> reports/subject_coadaptation_r8/launcher.log
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/subject_coadapt_ceiling_r8.py \
    --anatomy $anatomy --volumes $volumes --split block --reverse \
    --arms base,mask,mask_in,param,param_in,param_ctrl,param_h,grid,grid_in,grid_ctrl,grid_h --tag param > $log 2>&1
  rc=$?
  echo "done param_family helper $anatomy $volumes gpu$gpu rc=$rc" >> reports/subject_coadaptation_r8/launcher.log
done
