# Block-split ceiling with every arm that tests one mask + one VarNet per subject:
#   bash analysis/subject_coadapt/run_ceiling_block_full.sh <gpu> <anatomy> <start:stop>
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
gpu=$1; anatomy=$2; volumes=$3
log=analysis/subject_coadapt/ceiling_blockfull_${anatomy}_${volumes/:/-}.log
echo "start blockfull $anatomy $volumes gpu$gpu $(date)" >> analysis/subject_coadapt/launcher.log
CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/subject_coadapt_ceiling_r8.py --anatomy $anatomy --volumes $volumes \
  --split block --arms base,mask_in,recon,joint,recon_generic --eval-at 25,50,100,200,400 --checkpoint-cascades \
  --tag full > $log 2>&1
rc=$?
echo "done blockfull $anatomy $volumes gpu$gpu rc=$rc $(date)" >> analysis/subject_coadapt/launcher.log
