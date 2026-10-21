# Generic continued-training control, block split, fresh vs resumed Adam:
#   bash analysis/subject_coadapt/run_generic_ctrl.sh <gpu> <fresh|adam>
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
gpu=$1; mode=$2
extra=(); [ "$mode" = adam ] && extra=(--resume-adam)
for spec in "brain 1:2" "knee 0:1"; do
  read -r anatomy volumes <<< "$spec"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/subject_coadapt_ceiling_r8.py --anatomy $anatomy --volumes $volumes \
    --split block --arms base,recon,recon_generic --eval-at 25,50,100,200,400 --checkpoint-cascades --tag $mode "${extra[@]}" \
    >> analysis/subject_coadapt/generic_ctrl_$mode.log 2>&1
  echo "done generic $mode $anatomy rc=$? $(date)" >> analysis/subject_coadapt/launcher.log
done
