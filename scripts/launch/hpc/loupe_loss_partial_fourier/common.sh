# Settings for the loupe_loss_partial_fourier CoreHPC jobs (reports/loupe_loss_partial_fourier/proposal.md). Sourced by
# the sbatch scripts, never run on the login node. Every arm is fscr1's tier-F LOUPE recipe (multistep_adaptivity's
# flair_arm_args) at seed 42 or 43, with the loss and the map's symmetry set:
#   Lrss_1_s<s>_tF       RSS L1, free map (the control, rerun at this run state)
#   Lcplx_1_s<s>_tF      complex-image L1 through the model's own detached maps (loss.loss_type=complex_l1)
#   Lrsssym_1_s<s>_tF    RSS L1, point-symmetric map (sampler.loupe_point_symmetric=true)
#   Lcplxsym_1_s<s>_tF   complex-image L1, point-symmetric map

EXP_DIR="${EXP_DIR:-$HOME/scratch/loupe_loss_partial_fourier}"
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

loss_pf_arm_args() {
    local arm="$1"
    if [[ ! "$arm" =~ ^L(rss|cplx)(sym)?_1_s(42|43)_tF$ ]]; then
        echo "unknown loupe_loss_partial_fourier arm '$arm'" >&2
        exit 1
    fi
    local loss="${BASH_REMATCH[1]}" symmetric="${BASH_REMATCH[2]}" seed="${BASH_REMATCH[3]}"
    flair_arm_args "$arm" L 1 "$seed" 1 F
    [[ "$loss" == cplx ]] && set_arg loss.loss_type=complex_l1
    [[ -n "$symmetric" ]] && set_arg sampler.loupe_point_symmetric=true
    return 0
}
