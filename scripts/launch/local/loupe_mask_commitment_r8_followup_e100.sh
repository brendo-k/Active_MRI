#!/bin/bash
# Local copy of followup.sbatch tasks 6-9 (mixdecay s42/s43/s44 and control s44, 100 epochs, step at 90), moved off CoreHPC
# while its queue was full. Same overrides; data from /export/scratch/bkadota/fastmri_raw. Two GPU lanes run in parallel,
# two runs each. Round 1's control_s42/s43 are linked into runs_e100 so the eval pairs every mixdecay seed.
#
#   nohup bash scripts/launch/local/loupe_mask_commitment_r8_followup_e100.sh > /export/scratch/bkadota/loupe_mask_commitment_r8/local_e100.log 2>&1 &
set -o pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP_DIR=/export/scratch/bkadota/loupe_mask_commitment_r8
RUNS_DIR="$EXP_DIR/runs_e100"
mkdir -p "$RUNS_DIR"
for seed in 42 43; do
    [[ -e "$RUNS_DIR/control_s$seed" ]] || ln -s "$EXP_DIR/runs/control_s$seed" "$RUNS_DIR/control_s$seed"
done

run() {
    local gpu=$1 arm=$2 seed=$3
    local out="$RUNS_DIR/${arm}_s$seed" overrides=()
    if [[ "$arm" == mixdecay ]]; then
        overrides=(sampler.flat_mask_fraction=0.25 sampler.flat_mask_decay_epochs=30)
    fi
    if [[ -e "$out/last_model.pt" ]]; then
        mv "$out" "$out.attempt-$(date +%Y%m%d-%H%M%S)"
    fi
    mkdir -p "$out"
    echo "$(date -Is) start gpu=$gpu $arm s$seed"
    (cd "$REPO" && source .venv/bin/activate && \
     CUDA_VISIBLE_DEVICES=$gpu WANDB_RUN_GROUP=loupe_mask_commitment_r8_runs_e100 OMP_NUM_THREADS=4 \
     HDF5_USE_FILE_LOCKING=FALSE PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python train.py --model sequential recon=varnet recon.num_cascades=12 \
        datasets=fastmri_local_v1_flair_knee datasets.limit_volumes=1 \
        datasets.roots.local_batch2=/export/scratch/bkadota/fastmri_raw/knee \
        'misc.resolution=[320,320]' \
        'sampler.accelerations=[8.0]' sampler.budget_over=acquired \
        sampler.sampler_type=loupe_2d sampler.loupe_vd_init_power=0 sampler.anatomy_conditioning=false \
        sampler.lr=0.001 \
        training.num_step=1 training.batch_size=16 training.num_epochs=100 \
        training.lr=0.001 training.lr_scheduler=step training.lr_step_size=90 training.lr_gamma=0.1 \
        training.val_every=5 training.seed="$seed" \
        "${overrides[@]}" \
        "output.out_dir=$out" "output.run_name=lmc_fu_local_${arm}_s$seed" > "$out.stdout.log" 2>&1)
    echo "$(date -Is) end gpu=$gpu $arm s$seed rc=$?"
}

lane() {
    local gpu=$1; shift
    while (( $# )); do
        run "$gpu" "$1" "$2"
        shift 2
    done
}

lane 0 mixdecay 42 mixdecay 44 &
lane 1 mixdecay 43 control 44 &
wait
echo "$(date -Is) all done"
