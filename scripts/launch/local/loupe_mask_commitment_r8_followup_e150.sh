#!/bin/bash
# Local copy of followup.sbatch tasks 0, 1, 3, 4 (control_e150 and mix_e150, seeds 42/43; 150 epochs, step at 135), moved
# off CoreHPC while its queue was full. Tasks 2 and 5 (seed 44) stay on CoreHPC, so each control/mix pair trains on one
# platform. Same overrides as the sbatch; data from /export/scratch/bkadota/fastmri_raw. One run per GPU, GPUs 0-3.
#
#   nohup bash scripts/launch/local/loupe_mask_commitment_r8_followup_e150.sh > /export/scratch/bkadota/loupe_mask_commitment_r8/local_e150.log 2>&1 &
set -o pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
RUNS_DIR=/export/scratch/bkadota/loupe_mask_commitment_r8/runs_e150
mkdir -p "$RUNS_DIR"

run() {
    local gpu=$1 arm=$2 seed=$3
    local out="$RUNS_DIR/${arm}_s$seed" overrides=()
    if [[ "$arm" == mix ]]; then
        overrides=(sampler.flat_mask_fraction=0.25)
    fi
    if [[ -e "$out/last_model.pt" ]]; then
        mv "$out" "$out.attempt-$(date +%Y%m%d-%H%M%S)"
    fi
    mkdir -p "$out"
    echo "$(date -Is) start gpu=$gpu $arm s$seed"
    (cd "$REPO" && source .venv/bin/activate && \
     CUDA_VISIBLE_DEVICES=$gpu WANDB_RUN_GROUP=loupe_mask_commitment_r8_runs_e150 OMP_NUM_THREADS=4 \
     HDF5_USE_FILE_LOCKING=FALSE PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python train.py --model sequential recon=varnet recon.num_cascades=12 \
        datasets=fastmri_local_v1_flair_knee datasets.limit_volumes=1 \
        datasets.roots.local_batch2=/export/scratch/bkadota/fastmri_raw/knee \
        'misc.resolution=[320,320]' \
        'sampler.accelerations=[8.0]' sampler.budget_over=acquired \
        sampler.sampler_type=loupe_2d sampler.loupe_vd_init_power=0 sampler.anatomy_conditioning=false \
        sampler.lr=0.001 \
        training.num_step=1 training.batch_size=16 training.num_epochs=150 \
        training.lr=0.001 training.lr_scheduler=step training.lr_step_size=135 training.lr_gamma=0.1 \
        training.val_every=5 training.seed="$seed" \
        "${overrides[@]}" \
        "output.out_dir=$out" "output.run_name=lmc_fu_local_${arm}_e150_s$seed" > "$out.stdout.log" 2>&1)
    echo "$(date -Is) end gpu=$gpu $arm s$seed rc=$?"
}

run 0 control 42 &
run 1 control 43 &
run 2 mix 42 &
run 3 mix 43 &
wait
echo "$(date -Is) all done"
