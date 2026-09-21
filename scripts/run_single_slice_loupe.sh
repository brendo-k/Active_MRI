#!/usr/bin/env bash
# Zero-step LOUPE overfit on single slices, to compare the learned patterns:
#   brain s8 vs knee s18   -> does anatomy change the pattern?
#   brain s8 vs brain s4   -> does the slice within a volume change the pattern?
#   brain s8 vs brain s8'  -> how much difference is just optimization noise?
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

OUT=${OUT:-checkpoints/single_slice_loupe_converged}
ITERS=${ITERS:-3000}
# The mask parameter needs a much larger step than the reconstructor, and a
# neutral p=0.5 init, or the learned pattern just echoes its initialization.
MASK_LR=${MASK_LR:-0.05}
MASK_INIT=${MASK_INIT:-constant}
BRAIN_FILE=file_brain_AXT2_200_2000020.h5
KNEE_FILE=file1000010.h5

mkdir -p "$OUT"

run() {  # name gpu anatomy file slice seed sampler
    local name=$1 gpu=$2 anatomy=$3 file=$4 slice=$5 seed=$6 sampler=$7
    CUDA_VISIBLE_DEVICES=$gpu python scripts/train_single_slice_loupe.py \
        --anatomy "$anatomy" --file-name "$file" --slice-id "$slice" \
        --sampler-type "$sampler" --seed "$seed" --iters "$ITERS" \
        --mask-lr "$MASK_LR" --mask-init "$MASK_INIT" \
        --out-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
}

for sampler in loupe_2d loupe_1d; do
    tag=${sampler#loupe_}
    echo "=== $sampler ==="
    run "brain_s8_${tag}"      0 brain "$BRAIN_FILE" 8  42 "$sampler" &
    run "brain_s4_${tag}"      1 brain "$BRAIN_FILE" 4  42 "$sampler" &
    run "knee_s18_${tag}"      2 knee  "$KNEE_FILE"  18 42 "$sampler" &
    run "brain_s8_seed7_${tag}" 0 brain "$BRAIN_FILE" 8  7 "$sampler" &
    wait
done

echo "All runs complete. Outputs in $OUT"
