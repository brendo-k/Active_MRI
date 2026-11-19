# Settings for the seqmri_lr_bias_r8 CoreHPC jobs (reports/seqmri_lr_bias_r8/proposal.md). Sourced by the sbatch
# scripts, never run on the login node. Every arm is an fscr1 tier-F arm (multistep_adaptivity's flair_arm_args, the
# recipe the controls ran) with one setting replaced, so the recipes are not retyped:
#   <F|Fb|L>_<K>_s42_fscr1_tF_r150  the fscr1 run resumed from its epoch-100 checkpoint (optimizer state included) to
#                                   epoch 150, at the lr the step schedule gives after epoch 90 (1e-4), in a new run folder
#   <F|Fb>_4_s42_lrs_tF             the sampler in its own Adam group at sampler.lr=1e-4 (1e-5 after the step at 90)
#   <F|Fb>_4_s42_bias_tF            one learned logit map per step (sampler.seqmri_step_bias=true)

EXP_DIR="${EXP_DIR:-$HOME/scratch/seqmri_lr_bias_r8}"
FSCR1_RUNS="${FSCR1_RUNS:-$HOME/scratch/multistep_adaptivity/runs}"
source "$(dirname "${BASH_SOURCE[0]}")/../multistep_adaptivity/common.sh"

set_arg() {
    local pair="$1" key="${1%%=*}" i
    for i in "${!ARM_ARGS[@]}"; do
        if [[ "${ARM_ARGS[$i]}" == "$key="* ]]; then
            ARM_ARGS[i]="$pair"
            return
        fi
    done
    ARM_ARGS+=("$pair")
}

lr_bias_arm_args() {
    local arm="$1"
    if [[ "$arm" =~ ^((F|Fb|L)_[0-9]+_s42_fscr1_tF)_r150$ ]]; then
        local control="${BASH_REMATCH[1]}"
        arm_args "$control"
        set_arg "output.checkpoint=$FSCR1_RUNS/$control/last_model.pt"
        set_arg training.num_epochs=150
    elif [[ "$arm" =~ ^(F|Fb)_4_s42_(lrs|bias)_tF$ ]]; then
        local kind="${BASH_REMATCH[1]}" change="${BASH_REMATCH[2]}"
        arm_args "${kind}_4_s42_fscr1_tF"
        case "$change" in
            lrs) set_arg sampler.lr=1e-4 ;;
            bias) set_arg sampler.seqmri_step_bias=true ;;
        esac
    else
        echo "unknown seqmri_lr_bias_r8 arm '$arm'" >&2
        exit 1
    fi
    set_arg "output.run_name=$arm"
}
