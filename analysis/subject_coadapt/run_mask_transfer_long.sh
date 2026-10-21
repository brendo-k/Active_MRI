# Does a long-fitted per-volume mask carry within the subject? `mask` (fit on F only) at 1500 steps, scored on H,
# same block-split volumes as the 600/1500-step mask_in follow-ups:
#   bash analysis/subject_coadapt/run_mask_transfer_long.sh <gpu> <anatomy> <start:stop>
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
gpu=$1; anatomy=$2; volumes=$3
CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/subject_coadapt_ceiling_r8.py --anatomy $anatomy --volumes $volumes \
  --split block --arms base,mask --mask-steps 1500 --checkpoint-cascades --tag mask1500 \
  >> analysis/subject_coadapt/followup_mask1500_$anatomy.log 2>&1
echo "done mask1500 $anatomy rc=$? $(date)" >> analysis/subject_coadapt/launcher.log
