# Leak-free ECHO pilot (--echo-loss heldout), same slices as run_echo_pilot.sh:
#   bash analysis/subject_coadapt/run_echo_pilot_heldout.sh <gpu> <lr> <rho>
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
gpu=$1; lr=$2; rho=$3
tag=heldout_lr${lr}_rho${rho}
CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/subject_ssl_tta_r8.py --anatomy brain --volumes 1:2 --method echo \
  --arms base,ssl_slice --slice-count 2 --steps 500 --eval-at 25,50,100,200,500 --gamma 0 --freeze-sens \
  --echo-loss heldout --rho $rho --lr $lr --tag echo_$tag > analysis/subject_coadapt/echo_pilot_$tag.log 2>&1
echo "done echo $tag rc=$? $(date)" >> analysis/subject_coadapt/launcher.log
