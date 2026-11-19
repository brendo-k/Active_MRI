#!/bin/bash
# The fscr1 validation job (reports/multistep_adaptivity/addendum_2026-11-16_fscr1.md). It trains each fscr1 arm
# exactly as train.sbatch does, for 10% of one epoch, scores it with the eval train.sbatch runs (gradients included), and
# checks the twins' nulls, the signed-log arms' positive controls (a data swap and zeroed data channels both move their
# draws), the swap function's self-check, that no draw lands outside a slice's acquired region, and the shipped commit.
# The fscr1 training jobs are submitted with --dependency=afterok on this job, so they start only if every check
# passes. Submit from $EXP_DIR (see run.md section 5d):
#
#   sbatch --gres=gpu:nvidia_rtx_pro_6000:1 --export=ALL,GPU_NAME='RTX PRO 6000',EXPECTED_COMMIT=<shipped hash> \
#       code/scripts/launch/hpc/multistep_adaptivity/validate_fscr1.sbatch
#
#SBATCH --job-name=multistep_validate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=06:00:00
#SBATCH --output=slurm_log/%x-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brenden.kadota@ucsf.edu

set -uo pipefail
source "${EXP_DIR:-$HOME/scratch/multistep_adaptivity}/code/scripts/launch/hpc/multistep_adaptivity/common.sh"
: "${EXPECTED_COMMIT:?submit with --export=ALL,GPU_NAME=<card>,EXPECTED_COMMIT=<hash>}"
start_job
OUT="$EXP_DIR/validate/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
ARMS=(F_4_s42_fscr1_tF Fb_4_s42_fscr1_tF L_1_s42_fscr1_tF F_4_s42_fscr1_tS5 Fb_4_s42_fscr1_tS5 L_1_s42_fscr1_tS5)
CHECKS=()
declare -A RESULT

check() {
    local name="$1"
    shift
    if "$@" > "$OUT/$name.log" 2>&1; then RESULT[$name]=pass; else RESULT[$name]=FAIL; fi
    CHECKS+=("$name")
    echo "$(date +%H:%M:%S) $name: ${RESULT[$name]}"
}

shipped_commit() {
    git rev-parse HEAD
    [[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]]
}

smoke() {
    (arm_args "$1" && python train.py "${ARM_ARGS[@]}" "${RECIPE[@]}" "output.out_dir=$OUT/$1" training.num_epochs=1 \
        training.limit_batches=0.1)
}

score() {
    python scripts/eval/eval_multistep_adaptivity.py score --run "$OUT/$1" --name "$1" --out "$OUT/scores" --limit 120 \
        --no-wandb --gradients
}

nulls() {
    python - "$OUT/scores" "${ARMS[@]}" <<'EOF'
import json, sys
from pathlib import Path

scores = Path(sys.argv[1])
loaded = {name: json.loads((scores / f'{name}.json').read_text()) for name in sys.argv[2:]}
problems = []
for name, scored in loaded.items():
    guards = scored['guards']
    print(name, 'realised', round(scored['means']['realised_fraction']['all'], 4), 'most draws outside the acquired region',
          guards['most_draws_outside_the_acquired_region'], 'donors', scored['donors'])
    if not guards['no_draws_outside_the_acquired_region']:
        problems.append(f"{name}: up to {guards['most_draws_outside_the_acquired_region']} draws outside the acquired region")
    if name.startswith('L_'):
        continue
    if scored['counterfactual_with_own_data_reproduces_every_step'] is not True:
        problems.append(f'{name}: the swap function does not reproduce its own steps')
    moved = scored['data_swap']['moved_by_step']['all']
    print(name, 'data swap moved by step', moved, 'zero data moved by step',
          scored['data_swap'].get('zero_data_moved_by_step', {}).get('all'))
    if name.startswith('Fb_'):
        cost = scored['swap_cost']['psnr']['all']
        if scored['swap_null']['donor_final_mask_equals_own'] != 1.0:
            problems.append(f"{name}: the donor's final mask equals its own in "
                            f"{scored['swap_null']['donor_final_mask_equals_own']} of slices")
        if any(value != 0 for value in moved):
            problems.append(f'{name}: a data swap moved samples {moved}')
        if cost['mean'] != 0 or cost['ci95'] != [0.0, 0.0]:
            problems.append(f"{name}: swap cost {cost['mean']} {cost['ci95']}")
    else:
        if not any(value > 0 for value in moved):
            problems.append(f'{name}: a data swap moved nothing, so the positive control failed')
        if not any(value > 0 for value in scored['data_swap']['zero_data_moved_by_step']['all']):
            problems.append(f'{name}: zeroing its data channels moved nothing, so it reads no data')
print('\n'.join(problems) or 'twin nulls hold, the signed-log arms read their data, every arm reproduces its own steps '
      'and no draw lands outside the acquired region')
sys.exit(1 if problems else 0)
EOF
}

run_info_commits() {
    python - "$OUT" "$EXPECTED_COMMIT" <<'EOF'
import json, sys
from pathlib import Path

infos = {path.parent.name: json.loads(path.read_text()) for path in Path(sys.argv[1]).glob('*/run_info.json')}
bad = {name: (info['commit'], info['dirty_tracked_files']) for name, info in infos.items()
       if info['commit'] != sys.argv[2] or info['dirty_tracked_files']}
print(f'{len(infos)} runs recorded', {name: info['commit'] for name, info in infos.items()})
print(f'not the shipped commit, or dirty: {bad}' if bad else 'every run recorded the shipped commit, clean')
sys.exit(1 if bad or len(infos) != 6 else 0)
EOF
}

check commit shipped_commit
check tests python -m pytest -q tests
for arm in "${ARMS[@]}"; do
    check "smoke_$arm" smoke "$arm"
done
for arm in "${ARMS[@]}"; do
    check "score_$arm" score "$arm"
done
check nulls nulls
check run_info run_info_commits

echo
echo "Epoch lines (10% of a train epoch, compile included, then the full FLAIR dev val half):"
for arm in "${ARMS[@]}"; do
    echo "  $arm: $(grep -h 'Epoch = ' "$OUT/smoke_$arm.log" | tail -n 1 | sed 's/.*\(TrainTime.*\)/\1/')"
done
echo
failed=0
for name in "${CHECKS[@]}"; do
    printf '  %-24s %s\n' "$name" "${RESULT[$name]}"
    [[ "${RESULT[$name]}" == pass ]] || failed=1
done
echo "logs: $OUT"
exit "$failed"
