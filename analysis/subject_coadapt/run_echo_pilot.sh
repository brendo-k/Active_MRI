# ECHO test-time adaptation pilot (scripts/diagnostics/subject_ssl_tta_r8.py --method echo), per-slice on brain volume 1:
#   bash analysis/subject_coadapt/run_echo_pilot.sh <gpu>
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
gpu=$1
run() {
  tag=$1; shift
  CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/subject_ssl_tta_r8.py --anatomy brain --volumes 1:2 --method echo \
    --arms base,ssl_slice --slice-count 2 --steps 200 --eval-at 25,50,100,200 --gamma 0 --freeze-sens --tag echo_$tag "$@" \
    > analysis/subject_coadapt/echo_pilot_$tag.log 2>&1
  echo "done echo $tag rc=$? $(date)" >> analysis/subject_coadapt/launcher.log
}
run acquired_lr1e-5 --echo-loss acquired --lr 1e-5
run unseen_lr1e-5 --echo-loss unseen --lr 1e-5
run acquired_lr3e-5 --echo-loss acquired --lr 3e-5
run unseen_lr3e-5 --echo-loss unseen --lr 3e-5
