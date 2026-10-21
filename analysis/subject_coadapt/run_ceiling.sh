# Per-subject co-adaptation ceiling (scripts/diagnostics/subject_coadapt_ceiling_r8.py), one shard per call:
#   bash analysis/subject_coadapt/run_ceiling.sh <gpu> <anatomy> <start:stop>
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
gpu=$1; anatomy=$2; volumes=$3
log=analysis/subject_coadapt/ceiling_${anatomy}_${volumes/:/-}.log
echo "start $anatomy $volumes gpu$gpu $(date)" >> analysis/subject_coadapt/launcher.log
CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/subject_coadapt_ceiling_r8.py --anatomy $anatomy --volumes $volumes \
  --eval-at 50,100,200,400 > $log 2>&1
rc=$?
echo "done $anatomy $volumes gpu$gpu rc=$rc $(date)" >> analysis/subject_coadapt/launcher.log
