#!/bin/bash
# Path / estimator test for the 3-mask CG-SENSE R8 run: PDFS row toward the brain mask, brain row toward the shared
# mask as the control, along the full and the radial-density direction (scripts/diagnostics/mask_path_gradient_r8.py).
cd "$(dirname "$0")/../../.." || exit 1
source .venv/bin/activate
OUT=analysis/anatcond_cgsense/metric_apodized
run() {
  local gpu=$1 cls=$2 target=$3 direction=$4 per_batch=$5 stride=$6
  CUDA_VISIBLE_DEVICES=$gpu python scripts/diagnostics/mask_path_gradient_r8.py --cls "$cls" --target "$target" \
    --direction "$direction" --per-batch "$per_batch" --stride "$stride" \
    > "$OUT/mask_path_${cls}_to_${target}_${direction}.log" 2>&1
  echo "$cls -> $target ($direction) exit $?" >> "$OUT/mask_path_status.log"
}
run 0 knee-PDFS brain radial 4.17 1 &
run 1 knee-PDFS brain full 4.17 1 &
run 2 brain shared radial 8 2 &
run 3 brain shared full 8 2 &
wait
