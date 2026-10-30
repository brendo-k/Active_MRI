#!/usr/bin/env bash
# Local share of the loss-domain runs (reports/loss_domain/proposal.md): seed 42 on two GPUs, one chain per GPU.
#   GPU_A: rss, then kspace_rss        GPU_B: complex, then kspace_rss_M
# Each run waits for its GPU to be idle. A run that runs out of GPU memory is rerun with
# recon.checkpoint_cascades=true; any other stop is rerun once from scratch, keeping the failed log.
# Run it from a code snapshot (a detached worktree at the launch commit) so later edits cannot reach the runs:
#   nohup setsid <snapshot>/scripts/launch/queue_loss_domain_local.sh > <log> 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."
PY="${PY:-/home/bkadota/code/active_mri/SeqMRI/.venv/bin/python}"
export PATH="$(dirname "$PY"):$PATH"
GREP=/usr/bin/grep
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
OUT_ROOT="${OUT_ROOT:-$HOME/data/seqmri_checkpoints/loss_domain/runs}"
LAST_EPOCH="Epoch = \[ *$EPOCHS/ *$EPOCHS\]"
mkdir -p "$OUT_ROOT"

stamp() { date '+%F %T'; }
run_dir() { echo "$OUT_ROOT/loss_domain_${1}_r8_flair_knee_${EPOCHS}ep_seed${SEED}"; }
gpu_busy() { [[ -n "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader)" ]]; }

train() {
    local arm=$1 gpu=$2 out extra=() rc
    out="$(run_dir "$arm")"
    if $GREP -qE "$LAST_EPOCH" "$out.log" 2>/dev/null; then
        echo "$(stamp) $arm already finished; not retrained"
        return 0
    fi
    for attempt in 1 2 3; do
        while gpu_busy "$gpu"; do sleep 120; done
        echo "$(stamp) start $arm seed $SEED GPU $gpu attempt $attempt ${extra[*]}"
        FOREGROUND=1 GPU="$gpu" ARM="$arm" SEED="$SEED" EPOCHS="$EPOCHS" OUT_ROOT="$OUT_ROOT" \
            scripts/launch/train_loss_domain_r8_local_v1.sh "${extra[@]}"
        rc=$?
        if $GREP -qE "$LAST_EPOCH" "$out.log"; then
            echo "$(stamp) done $arm (exit $rc)"
            return 0
        fi
        mv "$out.log" "$out.attempt$attempt.log"
        if $GREP -q 'out of memory' "$out.attempt$attempt.log"; then
            echo "$(stamp) $arm ran out of GPU memory; rerunning with recon.checkpoint_cascades=true"
            extra=(recon.checkpoint_cascades=true)
        else
            echo "$(stamp) $arm stopped early (exit $rc); log kept as $out.attempt$attempt.log"
            [[ $attempt -ge 2 ]] && return 1
        fi
        sleep 120
    done
    return 1
}

chain() {
    local gpu=$1; shift
    for arm in "$@"; do train "$arm" "$gpu"; done
}

echo "$(stamp) queue start: GPU $GPU_A rss -> kspace_rss, GPU $GPU_B complex -> kspace_rss_M, seed $SEED"
chain "$GPU_A" rss kspace_rss &
chain "$GPU_B" complex kspace_rss_M &
wait
echo "$(stamp) queue finished"
