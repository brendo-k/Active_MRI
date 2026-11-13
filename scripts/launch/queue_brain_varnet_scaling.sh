#!/usr/bin/env bash
# Queue for reports/brain_varnet_scaling (proposal.md): brain-only VarNets on the frozen shared mask S, over cascades x
# training volumes, all at 7,600 optimiser steps (100 epochs of 80 volumes). Val is brain only and pinned to the same
# 10 volumes / 150 slices in every run, so best_model.pt is the brain val peak.
#
#   volumes  limit_volumes  steps/epoch  epochs  lr step  val every
#   20       0.25           19           400     360      20
#   40       0.5            38           200     180      10
#   80       1.0            76           100     90       5
#
# Appends jobs to $Q/jobs.txt (longest first) and starts one worker per GPU in 0-1 unless NO_WORKERS=1.
#
#   bash scripts/launch/queue_brain_varnet_scaling.sh <code snapshot dir> <seed> <cascades:volumes> ...
#   bash scripts/launch/queue_brain_varnet_scaling.sh ../run-bvs-<hash> 42 all
set -euo pipefail
CODE="${1:?code snapshot dir}"
SEED="${2:?seed}"
shift 2
Q=/home/bkadota/data/seqmri_checkpoints/brain_varnet_scaling
S=/home/bkadota/data/seqmri_checkpoints/loss_domain/runs_v2/loss_domain_rss_r8_flair_knee_100ep_seed42/best_model.pt
DEADLINE="${DEADLINE:-none}"
declare -A LIMIT=([20]=0.25 [40]=0.5 [80]=1.0) EPOCHS=([20]=400 [40]=200 [80]=100) STEP=([20]=360 [40]=180 [80]=90) VAL=([20]=20 [40]=10 [80]=5)
# Minutes per run (7,600 steps + 20 brain vals) from the 2026-11-13 smoke test: 1.32 / 1.00 / 0.74 s/step at 12 / 6 / 2
# cascades on an A6000 (data loading sets a floor); 4 and 8 interpolated. Used only for ordering and DEADLINE.
declare -A MINUTES=([2]=97 [4]=113 [6]=130 [8]=145 [12]=170)
BASE="recon=varnet datasets=fastmri_local_v1_flair_knee misc.resolution=[320,320] sampler.accelerations=[8.0]
sampler.budget_over=acquired training.num_step=1 training.batch_size=16 sampler.sampler_type=loupe_2d
sampler.mask_train_mode=straight_through sampler.anatomy_conditioning=true datasets.train_anatomies=[brain]
datasets.val_anatomies=[brain] datasets.val_limit_volumes=0.25
sampler.fixed_mask_checkpoints={brain:$S,knee-PD:$S,knee-PDFS:$S}"
BASE=$(echo $BASE)

cells=("$@")
if [[ "${cells[*]}" == all ]]; then
    cells=()
    for v in 80 40 20; do for c in 12 8 6 4 2; do cells+=("$c:$v"); done; done
fi
mkdir -p "$Q/bin" "$Q/claims"
cp -n scripts/launch/brain_varnet_scaling_worker.sh "$Q/bin/worker.sh"
touch "$Q/jobs.txt"
lines=()
for cell in "${cells[@]}"; do
    c=${cell%:*}; v=${cell#*:}
    name="c${c}_v${v}_s$SEED"
    if /usr/bin/grep -q "^$name|" "$Q/jobs.txt"; then echo "$name is already queued" >&2; exit 1; fi
    minutes=${MINUTES[$c]}
    lines+=("$minutes $name|$CODE|$minutes|$BASE recon.num_cascades=$c datasets.limit_volumes=${LIMIT[$v]} training.num_epochs=${EPOCHS[$v]} training.lr_step_size=${STEP[$v]} training.val_every=${VAL[$v]} training.seed=$SEED")
done
{
    echo "# seed $SEED: ${cells[*]}"
    printf '%s\n' "${lines[@]}" | sort -rn -k1,1 | cut -d' ' -f2-
} >> "$Q/jobs.txt"
[[ "${NO_WORKERS:-0}" == 1 ]] && { echo "queued ${#lines[@]} jobs; workers not started"; exit 0; }
rm -f "$Q/STOP"
for gpu in 0 1; do
    QUEUE_DIR="$Q" GPU="$gpu" DEADLINE="$DEADLINE" RUN_GROUP=brain_varnet_scaling RUN_PREFIX=bvs_ \
        nohup setsid bash "$Q/bin/worker.sh" > "$Q/worker$gpu.s$SEED.log" 2>&1 &
    sleep 90
done
echo "queued ${#lines[@]} jobs; workers started"
