# Block-split ceiling, VarNet arms only (F = first half of each volume, H = second half after a 2-slice gap):
#   bash reports/subject_coadaptation_r8/run_ceiling_block.sh <gpu> <anatomy> <start:stop>
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
gpu=$1; anatomy=$2; volumes=$3
log=reports/subject_coadaptation_r8/ceiling_block_${anatomy}_${volumes/:/-}.log
echo "start block $anatomy $volumes gpu$gpu $(date)" >> reports/subject_coadaptation_r8/launcher.log
CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/subject_coadapt_ceiling_r8.py --anatomy $anatomy --volumes $volumes \
  --split block --arms base,recon,recon_ctrl --eval-at 50,100,200,400 --checkpoint-cascades > $log 2>&1
rc=$?
echo "done block $anatomy $volumes gpu$gpu rc=$rc $(date)" >> reports/subject_coadaptation_r8/launcher.log
