# ECHO pilot at higher lr and 1000 steps (same slices as run_echo_pilot.sh):
#   bash analysis/subject_coadapt/run_echo_pilot_highlr.sh <gpu> <lr>
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
gpu=$1; lr=$2
CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/subject_ssl_tta_r8.py --anatomy brain --volumes 1:2 --method echo \
  --arms base,ssl_slice --slice-count 2 --steps 1000 --eval-at 25,50,100,200,500,1000 --gamma 0 --freeze-sens \
  --echo-loss acquired --lr $lr --tag echo_acquired_lr${lr}_1000 \
  > analysis/subject_coadapt/echo_pilot_acquired_lr${lr}_1000.log 2>&1
echo "done echo acquired lr$lr 1000 rc=$? $(date)" >> analysis/subject_coadapt/launcher.log
