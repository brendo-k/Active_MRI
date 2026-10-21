# Follow-ups to the per-subject report:
#   maskin <gpu>: block split, mask_in at 600 steps, then the VarNet fine-tuned on F with that mask held fixed
#   echovol <gpu>: ECHO (unseen, lr 1e-5) fitted per volume on every slice's acquired data
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
mode=$1; gpu=$2
if [ "$mode" = maskin ]; then
  for spec in "brain 1:5" "knee 0:3"; do
    read -r anatomy volumes <<< "$spec"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/subject_coadapt_ceiling_r8.py --anatomy $anatomy --volumes $volumes \
      --split block --arms base,mask_in,recon,recon_on_mask_in --mask-steps 600 --eval-at 25,50,100 --checkpoint-cascades \
      --tag maskin600 >> reports/subject_coadaptation_r8/followup_maskin.log 2>&1
    echo "done maskin $anatomy rc=$? $(date)" >> reports/subject_coadaptation_r8/launcher.log
  done
else
  for spec in "brain 1:2" "knee 0:1"; do
    read -r anatomy volumes <<< "$spec"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/subject_ssl_tta_r8.py --anatomy $anatomy --volumes $volumes --method echo \
      --echo-loss unseen --lr 1e-5 --gamma 0 --freeze-sens --arms base,ssl_vol --steps 200 --eval-at 25,50,100,200 \
      --checkpoint-cascades --tag echo_vol_unseen_lr1e-5 >> reports/subject_coadaptation_r8/followup_echovol.log 2>&1
    echo "done echovol $anatomy rc=$? $(date)" >> reports/subject_coadaptation_r8/launcher.log
  done
fi
