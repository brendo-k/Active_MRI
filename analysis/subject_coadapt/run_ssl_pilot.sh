# ZS-SSL pilot grid (scripts/diagnostics/subject_ssl_tta_r8.py) on one brain and one knee volume:
#   bash analysis/subject_coadapt/run_ssl_pilot.sh <gpu>
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
gpu=$1
run() {
  tag=$1; shift
  extra=("$@")
  for spec in "brain 1:2" "knee 0:1"; do
    read -r anatomy volumes <<< "$spec"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/subject_ssl_tta_r8.py --anatomy $anatomy --volumes $volumes \
      --arms base,ssl_vol,ssl_slice --slice-count 2 --steps 200 --eval-at 25,50,100,200 --batch 2 --tag pilot_$tag "${extra[@]}" \
      >> analysis/subject_coadapt/ssl_pilot_$tag.log 2>&1
    echo "done $tag $anatomy rc=$? $(date)" >> analysis/subject_coadapt/launcher.log
  done
}
run lr1e-5_rho0.4_sens --lr 1e-5 --rho 0.4
run lr1e-5_rho0.4 --lr 1e-5 --rho 0.4 --freeze-sens
run lr3e-6_rho0.4 --lr 3e-6 --rho 0.4 --freeze-sens
run lr1e-5_rho0.2 --lr 1e-5 --rho 0.2 --freeze-sens
run lr3e-5_rho0.4 --lr 3e-5 --rho 0.4 --freeze-sens
