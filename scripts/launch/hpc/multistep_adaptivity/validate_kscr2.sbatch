#!/bin/bash
# The kscr2 validation job (reports/multistep_adaptivity/addendum_2026-11-15.md). It trains each kscr2 arm exactly as
# train.sbatch does, for 10% of one epoch, scores it with the eval train.sbatch runs (gradients included), and checks the
# twin's nulls, the adaptive arms' positive control, the swap function's self-check and the shipped commit. The kscr2
# training jobs are submitted with --dependency=afterok on this job, so they start only if every check passes. Submit
# from $EXP_DIR (see run.md section 5c):
#
#   sbatch --gres=gpu:nvidia_rtx_pro_6000:1 --export=ALL,GPU_NAME='RTX PRO 6000',EXPECTED_COMMIT=<shipped hash> \
#       code/scripts/launch/hpc/multistep_adaptivity/validate_kscr2.sbatch
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
ARMS=(T_16_s42_kscr2 A_16_s42_kscr2 C_16_s42_kscr2 S_16_s42_kscr2 T_1_s42_kscr2)
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
    python - "$OUT/scores" <<'EOF'
import json, sys
from pathlib import Path

scores = Path(sys.argv[1])
loaded = {name: json.loads((scores / f'{name}.json').read_text())
          for name in ('T_16_s42_kscr2', 'A_16_s42_kscr2', 'C_16_s42_kscr2', 'S_16_s42_kscr2')}
twin = loaded['T_16_s42_kscr2']
problems = []
if twin['swap_null']['donor_final_mask_equals_own'] != 1.0:
    problems.append(f"twin: donor's final mask equals its own in {twin['swap_null']['donor_final_mask_equals_own']} of slices")
if any(moved != 0 for moved in twin['data_swap']['moved_by_step']['all']):
    problems.append(f"twin: a data swap moved samples {twin['data_swap']['moved_by_step']['all']}")
cost = twin['swap_cost']['psnr']['all']
if cost['mean'] != 0 or cost['ci95'] != [0.0, 0.0]:
    problems.append(f"twin: swap cost {cost['mean']} {cost['ci95']}")
for name, scored in loaded.items():
    if scored['counterfactual_with_own_data_reproduces_every_step'] is not True:
        problems.append(f'{name}: the swap function does not reproduce its own steps')
for name in ('C_16_s42_kscr2', 'S_16_s42_kscr2'):
    moved = loaded[name]['data_swap']['moved_by_step']['all']
    print(name, 'moved by step', moved)
    if not any(value > 0 for value in moved):
        problems.append(f'{name}: a data swap moved nothing, so the positive control failed')
print('twin swap null', twin['swap_null'], 'twin moved', twin['data_swap']['moved_by_step']['all'], 'twin cost', cost)
print('\n'.join(problems) or 'nulls hold, every arm reproduces its own steps, and the positive controls are above 0')
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
sys.exit(1 if bad or len(infos) != 5 else 0)
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
echo "Epoch lines (10% of a train epoch, compile included, then the full dev val half):"
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
