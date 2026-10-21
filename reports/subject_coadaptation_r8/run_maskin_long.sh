# Per-volume mask oracle (frozen VarNet) at 1500 steps, same block-split volumes as the 600-step follow-up:
#   bash reports/subject_coadaptation_r8/run_maskin_long.sh <gpu>
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
gpu=$1
for spec in "brain 1:5" "knee 0:3"; do
  read -r anatomy volumes <<< "$spec"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/subject_coadapt_ceiling_r8.py --anatomy $anatomy --volumes $volumes \
    --split block --arms base,mask_in --mask-steps 1500 --checkpoint-cascades --tag maskin1500 \
    >> reports/subject_coadaptation_r8/followup_maskin1500.log 2>&1
  echo "done maskin1500 $anatomy rc=$? $(date)" >> reports/subject_coadaptation_r8/launcher.log
done
