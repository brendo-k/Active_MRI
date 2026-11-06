#!/usr/bin/env bash
# Scores each finished run against the 100-epoch reference as soon as a GPU can take it, and keeps the per-slice cache in
# $Q/eval, so the time before 08:00 only has the last arm left to score. ref_s42 is listed first because the first run
# sets the val loader, and every run must be scored at one batch size to see the same mask draws.
# Only an idle GPU is used, because sharing one doubled e30_bs8_s42's epoch time. Scoring waits for the bench, which needs its
# GPU to itself and could otherwise start in the gap after the arm before it. From 06:45 it no longer waits.
Q=/home/bkadota/data/seqmri_checkpoints/muon_layer_speed
CODE=/home/bkadota/code/active_mri/SeqMRI-worktrees/run-mls-0920f8b
PY=/home/bkadota/code/active_mri/SeqMRI/.venv/bin/python
stop_waiting_for_bench=$(date -d '2026-11-04 06:45' +%s)
stop_at=$(date -d '2026-11-04 07:45' +%s)
scored="$Q/eval/scored.txt"
mkdir -p "$Q/eval"
touch "$scored"

stamp() { date '+%F %T'; }
idle() { [[ -z "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader)" ]]; }

pick_gpu() {
    [[ -e "$Q/claims/bench/done" || -e "$Q/claims/bench/gave_up" ]] || (( $(date +%s) >= stop_waiting_for_bench )) || return
    for gpu in 2 3; do idle "$gpu" && { echo "$gpu"; return; }; done
}

finished_runs() {
    local runs=ref_s42 claim name arm seed link
    for claim in "$Q"/claims/*; do
        name=$(basename "$claim")
        [[ -e "$claim/done" && "$name" == *_s4? && -e "$Q/$name/last_model.pt" ]] || continue
        runs+=" $name"
        [[ -e "$Q/$name/ema_model.pt" ]] || continue
        arm=${name%_s4?}; seed=${name##*_s}; link="$Q/${arm}_ema_s$seed"
        mkdir -p "$link"
        ln -sfn "$Q/$name/ema_model.pt" "$link/last_model.pt"
        ln -sfn "$Q/$name/.hydra" "$link/.hydra"
        ln -sfn "$Q/$name.log" "$link.log"
        runs+=" ${arm}_ema_s$seed"
    done
    echo "$runs"
}

while (( $(date +%s) < stop_at )); do
    runs=$(finished_runs)
    if [[ "$runs" != "$(cat "$scored")" ]]; then
        gpu=$(pick_gpu)
        if [[ -n "$gpu" ]]; then
            echo "$(stamp) scoring on GPU $gpu: $runs"
            (cd "$CODE" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$CODE exec "$PY" scripts/eval/varnet_train_speed_eval.py \
                --queue-dir "$Q" --runs $runs --control ref --out "$Q/eval/eval.json")
            rc=$?
            echo "$(stamp) scoring rc=$rc"
            if (( rc == 0 )); then echo "$runs" > "$scored"; else sleep 240; fi
        fi
    fi
    sleep 60
done
echo "$(stamp) rolling eval stop"
