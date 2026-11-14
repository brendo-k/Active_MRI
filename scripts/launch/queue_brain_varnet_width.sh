#!/usr/bin/env bash
# Queue for reports/brain_varnet_width (proposal.md): brain-only VarNets on the frozen shared mask S over cascades x
# U-Net channels, all on the 80 tier-S training volumes for 100 epochs (7,600 optimiser steps), val every 5 epochs on
# the pinned 10-volume brain subset, early-stop patience 5. 12 cascades at 26 or 36 channels recompute their cascades
# in backward (recon.checkpoint_cascades=true) to fit in 48 GB; that changes speed, not results.
#
# Appends jobs to $Q/jobs.txt (longest first) and starts one worker per GPU in $GPUS (default 0 1 2 3) unless
# NO_WORKERS=1. Workers exit once every job line is claimed.
#
#   bash scripts/launch/queue_brain_varnet_width.sh <code snapshot dir> <seed> <cascades:chans> ...
#   bash scripts/launch/queue_brain_varnet_width.sh ../run-bvw-<hash> 42 all
set -euo pipefail
CODE="${1:?code snapshot dir}"
SEED="${2:?seed}"
shift 2
Q="${Q:-/home/bkadota/data/seqmri_checkpoints/brain_varnet_width}"
S=/home/bkadota/data/seqmri_checkpoints/loss_domain/runs_v2/loss_domain_rss_r8_flair_knee_100ep_seed42/best_model.pt
DEADLINE="${DEADLINE:-none}"
PATIENCE="${PATIENCE:-5}"
GPUS="${GPUS:-0 1 2 3}"
# Minutes per run (7,600 steps + 20 brain vals), scaled from the 2026-11-14 benchmark against 12 x 18 (1.21 s/step in
# brain_varnet_scaling). Used only for ordering and DEADLINE.
declare -A MINUTES=([3:12]=95 [3:18]=100 [3:26]=105 [3:36]=110 [6:12]=110 [6:18]=130 [6:26]=125 [6:36]=140
                    [12:12]=150 [12:18]=170 [12:26]=235 [12:36]=280)
BASE="recon=varnet datasets=fastmri_local_v1_flair_knee misc.resolution=[320,320] sampler.accelerations=[8.0]
sampler.budget_over=acquired training.num_step=1 training.batch_size=16 sampler.sampler_type=loupe_2d
sampler.mask_train_mode=straight_through sampler.anatomy_conditioning=true datasets.train_anatomies=[brain]
datasets.val_anatomies=[brain] datasets.val_limit_volumes=0.25 training.early_stop_patience=$PATIENCE
datasets.limit_volumes=1.0 training.num_epochs=100 training.lr_step_size=90 training.val_every=5
sampler.fixed_mask_checkpoints={brain:$S,knee-PD:$S,knee-PDFS:$S}"
BASE=$(echo $BASE)

cells=("$@")
if [[ "${cells[*]}" == all ]]; then
    cells=(3:12 3:18 3:26 3:36 6:12 6:26 6:36 12:12 12:26 12:36)
fi
mkdir -p "$Q/bin" "$Q/claims"
cp -n scripts/launch/brain_varnet_width_worker.sh "$Q/bin/worker.sh"
touch "$Q/jobs.txt"
lines=()
for cell in "${cells[@]}"; do
    c=${cell%:*}; k=${cell#*:}
    name="c${c}_ch${k}_s$SEED"
    if /usr/bin/grep -q "^$name|" "$Q/jobs.txt"; then echo "$name is already queued" >&2; exit 1; fi
    minutes=${MINUTES[$cell]:?no minutes estimate for $cell}
    extra=""
    (( c == 12 && k >= 26 )) && extra=" recon.checkpoint_cascades=true"
    lines+=("$minutes $name|$CODE|$minutes|$BASE recon.num_cascades=$c recon.chans=$k training.seed=$SEED$extra")
done
{
    echo "# seed $SEED: ${cells[*]}"
    printf '%s\n' "${lines[@]}" | sort -rn -k1,1 | cut -d' ' -f2-
} >> "$Q/jobs.txt"
[[ "${NO_WORKERS:-0}" == 1 ]] && { echo "queued ${#lines[@]} jobs; workers not started"; exit 0; }
rm -f "$Q/STOP"
for gpu in $GPUS; do
    QUEUE_DIR="$Q" GPU="$gpu" DEADLINE="$DEADLINE" RUN_GROUP=brain_varnet_width RUN_PREFIX=bvw_ \
        nohup setsid bash "$Q/bin/worker.sh" > "$Q/worker$gpu.s$SEED.log" 2>&1 &
    sleep 90
done
echo "queued ${#lines[@]} jobs; workers started"
