# Settings shared by the multistep_adaptivity CoreHPC jobs (reports/multistep_adaptivity/proposal.md, sections 4 and 9).
# Sourced by the sbatch scripts, never run on the login node.
#
# Every run and eval of the experiment uses one GPU type. Submit with that type's gres, and set GPU_NAME to its
# CUDA device name (or a unique part of it, e.g. 'H100 NVL'); a job that lands on any other card exits before
# it starts.
#
# LAB_SERVER_TEST=1 runs the same scripts on the lab server, against its own data copy and the active Python env, to
# test them before shipping.

EXP_DIR="${EXP_DIR:-$HOME/scratch/multistep_adaptivity}"
CODE_DIR="$EXP_DIR/code"
RUNS_DIR="$EXP_DIR/runs"
SCORES_DIR="$EXP_DIR/scores"
: "${GPU_NAME:?set GPU_NAME to (part of) the pinned card name torch reports, e.g. GPU_NAME='H100 NVL'}"

DATA_OVERRIDES=(
    'datasets.roots.hpc="~/scratch/data/fastmri/{anatomy}/data"'
    'datasets.roots.local_batch2="~/scratch/data/fastmri/knee/data"'
)
if [[ "${LAB_SERVER_TEST:-0}" == 1 ]]; then
    DATA_OVERRIDES=()
fi
RECIPE=(--recon varnet --dataset fastmri_local_v1_flair_knee --acceleration 8 --batch-size 16
        'misc.resolution=[320,320]' sampler.budget_over=acquired datasets.limit_volumes=1 "${DATA_OVERRIDES[@]}")

start_job() {
    export WANDB_MODE=offline WANDB_DIR="$EXP_DIR/wandb" OMP_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
    mkdir -p "$WANDB_DIR"
    if [[ "${LAB_SERVER_TEST:-0}" != 1 ]]; then
        source /mnt/software/ohpc/pub/miniforge3/25.11.0/etc/profile.d/conda.sh
        conda activate dl
    fi
    local card
    card="$(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    if [[ "$card" != *"$GPU_NAME"* ]]; then
        echo "this job got a '$card', not the pinned '$GPU_NAME'" >&2
        exit 1
    fi
    cd "$CODE_DIR"
    echo "host=$(hostname) gpu=$card commit=$(git rev-parse HEAD)" \
         "$(python -c 'import sys, torch; print(f"python={sys.version.split()[0]} torch={torch.__version__}")')"
}

# The train.py arguments of one arm (proposal section 4.4), in ARM_ARGS.
#   B_<s>               base, seed s
#   A_<K>_s<s>, T_<K>   adaptive arm and data-blind twin, fine-tuned 30 epochs from B_<s> with fine-tune seed s
#   T_<K>_s42_f142      noise-floor twin: base 42, fine-tune seed 142
#   A_16_s42_scr, T_    from-scratch pair, 100 epochs on the base recipe
#   <A|T>_<K>_s<s>_kscr<n>  knee-only from-scratch subproblem, iteration n (addendum_2026-11-11.md): uniform-noise
#                       LOUPE start, knee PD + PDFS only, validated on the dev half of the knee val volumes
#   <C|S>_<K>_s<s>_kscr<n>  from iteration 2 (addendum_2026-11-15.md): adaptive arms whose sampler reads the fully
#                       sampled k-space (C, the clairvoyant ceiling) or also the trained VarNet's in-loop reconstructions
#                       (S, SeqMRI-style); their control is T_<K>_s<s>_kscr<n>
#   A_4_s42_rec         recon-input ablation: the sampler also reads frozen B_42's reconstruction of y * M_i
#   A_16_s42_img        channel ablation: the sampler reads the image channel, not the k-space one
arm_args() {
    local arm="$1"
    if [[ "$arm" =~ ^B_([0-9]+)$ ]]; then
        ARM_ARGS=(--model loupe --seed "${BASH_REMATCH[1]}" training.num_step=1 training.num_epochs=100
                  training.lr_step_size=90 sampler.loupe_vd_init_power=4 sampler.anatomy_conditioning=false)
        return
    fi
    if [[ ! "$arm" =~ ^([ATCS])_([0-9]+)_s([0-9]+)(_f([0-9]+)|_scr|_kscr([0-9]+)|_rec|_img)?$ ]]; then
        echo "unknown arm '$arm'" >&2
        exit 1
    fi
    local kind="${BASH_REMATCH[1]}" steps="${BASH_REMATCH[2]}" base_seed="${BASH_REMATCH[3]}"
    local variant="${BASH_REMATCH[4]}" floor_seed="${BASH_REMATCH[5]}" iteration="${BASH_REMATCH[6]}"
    if [[ "$kind" == [CS] && ( "$variant" != _kscr* || "$iteration" -lt 2 ) ]]; then
        echo "$arm: C and S arms exist only in knee subproblem iterations 2 and later" >&2
        exit 1
    fi
    local blind=false
    [[ "$kind" == T ]] && blind=true
    ARM_ARGS=(--model sequential sampler.sampler_type=seq_adapt_2d sampler.anatomy_conditioning=true
              training.num_step="$steps" sampler.data_blind="$blind")
    if [[ "$variant" == _kscr* ]]; then
        case "$iteration" in
            1) ARM_ARGS+=(--seed "$base_seed" training.num_epochs=100 training.lr_step_size=90) ;;
            2) ARM_ARGS+=(--seed "$base_seed" training.num_epochs=100 training.lr_step_size=90 recon.num_cascades=6) ;;
            *) echo "knee subproblem iteration $iteration is not defined" >&2; exit 1 ;;
        esac
        case "$kind" in
            C) ARM_ARGS+=(sampler.input=full) ;;
            S) ARM_ARGS+=(sampler.input=seqmri) ;;
        esac
        RECIPE=("${RECIPE[@]/#fastmri_local_v1_flair_knee/fastmri_local_v1_knee}" datasets.val_half=dev)
        return
    fi
    if [[ "$variant" == _scr ]]; then
        ARM_ARGS+=(--seed "$base_seed" training.num_epochs=100 training.lr_step_size=90 sampler.loupe_vd_init_power=4)
        return
    fi
    local base="$RUNS_DIR/B_$base_seed/last_model.pt"
    if [[ ! -f "$base" ]]; then
        echo "$arm fine-tunes $base, which does not exist yet" >&2
        exit 1
    fi
    case "$variant" in
        _rec) ARM_ARGS+=(sampler.input=recon "sampler.input_recon_checkpoint=$base") ;;
        _img) ARM_ARGS+=('sampler.data_channels=[image]') ;;
    esac
    ARM_ARGS+=(--seed "${floor_seed:-$base_seed}" training.num_epochs=30 training.lr=1e-4 training.lr_scheduler=constant
               sampler.residual_lr=1e-4 "output.pretrained_recon=$base" "sampler.init_mask_checkpoint=$base"
               'training.save_epochs=[10]')
}
