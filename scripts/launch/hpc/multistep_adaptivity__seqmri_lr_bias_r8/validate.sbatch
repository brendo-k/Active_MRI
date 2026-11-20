#!/bin/bash
# The multistep_adaptivity__seqmri_lr_bias_r8 validation job (reports/multistep_adaptivity__seqmri_lr_bias_r8/
# proposal.md). Training jobs are submitted with --dependency=afterok on it, so they start only if every check passes:
# - the shipped commit, clean, and the test suite;
# - each resume arm resumes its fscr1 checkpoint for a quarter of epoch 101: every optimizer group (Muon included) must
#   log lr 1e-4, and its training loss (a mean over a quarter of the epoch) must stay within 10% of the control's
#   epoch-100 training loss, so a learning-rate reset or misassigned optimizer state fails here;
# - each new from-scratch arm trains for 10% of one epoch exactly as train.sbatch does and is scored with --limit 120:
#   realised fraction, no draw outside the acquired region, the twins' swap nulls, the arms reading their data;
# - every run recorded the shipped commit.
# Submit from $EXP_DIR (~/scratch/multistep_adaptivity__seqmri_lr_bias_r8):
#
#   sbatch --gres=gpu:nvidia_rtx_pro_6000:1 --export=ALL,GPU_NAME='RTX PRO 6000',EXPECTED_COMMIT=<shipped hash> \
#       code/scripts/launch/hpc/multistep_adaptivity__seqmri_lr_bias_r8/validate.sbatch
#
#SBATCH --job-name=seqmri_lrb_validate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=06:00:00
#SBATCH --output=slurm_log/%x-%j.log

set -uo pipefail
TOPIC=multistep_adaptivity__seqmri_lr_bias_r8
source "${EXP_DIR:-$HOME/scratch/$TOPIC}/code/scripts/launch/hpc/$TOPIC/common.sh"
: "${EXPECTED_COMMIT:?submit with --export=ALL,GPU_NAME=<card>,EXPECTED_COMMIT=<hash>}"
start_job
OUT="$EXP_DIR/validate/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
read -r -a RESUME_ARMS <<< "${RESUME_ARMS-F_4_s42_fscr1_tF_r150 Fb_4_s42_fscr1_tF_r150 L_1_s42_fscr1_tF_r150}"
# The route arms (addendum_2026-11-19_route.md) are validated in their own EXP_DIR
# (~/scratch/multistep_adaptivity__seqmri_lr_bias_r8_rs2) with RESUME_ARMS='' and
# SMOKE_ARMS='F_4_s42_route_tF Fb_4_s42_route_tF F_4_s42_routebias_tF Fb_4_s42_routebias_tF'.
read -r -a SMOKE_ARMS <<< "${SMOKE_ARMS-F_4_s42_lrs_tF Fb_4_s42_lrs_tF}"
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

resume() {
    (lr_bias_arm_args "$1" && python train.py "${ARM_ARGS[@]}" "${RECIPE[@]}" "output.out_dir=$OUT/$1" \
        training.num_epochs=101 training.limit_batches=0.25)
}

resume_continues() {
    python - "$OUT/resume_$1.log" "$1" <<'EOF'
import re, sys
from pathlib import Path

log, arm = Path(sys.argv[1]).read_text(), sys.argv[2]
control = arm.removesuffix('_r150')
jobs = dict(line.split() for line in Path('reports/multistep_adaptivity/fscr1/fscr1_jobs.txt').read_text().splitlines()
            if line.strip())
pattern = re.compile(r'Epoch = \[\s*(\d+)/\s*\d+\] TrainLoss = ([-\d.eE+]+) .* LR = ([-\d.eE+,]+)')
resumed = [match for match in map(pattern.search, log.splitlines()) if match]
before = re.findall(r'Epoch = \[\s*100/\s*100\] TrainLoss = ([\d.eE+-]+)',
                    Path(f'reports/multistep_adaptivity/fscr1/multistep-{jobs[control]}.log').read_text())
problems = []
if len(resumed) != 1 or resumed[0].group(1) != '101':
    problems.append(f'expected one epoch line, epoch 101; got {[m.group(0)[:40] for m in resumed]}')
else:
    lrs = [float(value) for value in resumed[0].group(3).split(',')]
    loss, previous = float(resumed[0].group(2)), float(before[-1])
    print(f'{arm}: epoch 101 lr per group {lrs}, train loss {loss:.5f} (25% of the epoch) against {previous:.5f} at epoch 100')
    if any(abs(lr - 1e-4) > 1e-12 for lr in lrs):
        problems.append(f'lr per group {lrs}, expected 1e-4 everywhere')
    if abs(loss / previous - 1) > 0.10:
        problems.append(f'train loss {loss} is more than 10% from epoch 100 ({previous})')
print('\n'.join(problems) or 'resumes at 1e-4 in every group, and the loss continues')
sys.exit(1 if problems else 0)
EOF
}

smoke() {
    (lr_bias_arm_args "$1" && python train.py "${ARM_ARGS[@]}" "${RECIPE[@]}" "output.out_dir=$OUT/$1" \
        training.num_epochs=1 training.limit_batches=0.1)
}

score() {
    python scripts/eval/eval_multistep_adaptivity.py score --run "$OUT/$1" --name "$1" --out "$OUT/scores" --limit 120 \
        --no-wandb --gradients
}

nulls() {
    python - "$OUT/scores" ${SMOKE_ARMS[@]+"${SMOKE_ARMS[@]}"} <<'EOF'
import json, sys
from pathlib import Path

scores = Path(sys.argv[1])
problems = []
for name in sys.argv[2:]:
    scored = json.loads((scores / f'{name}.json').read_text())
    guards, moved = scored['guards'], scored['data_swap']['moved_by_step']['all']
    print(name, 'realised', round(scored['means']['realised_fraction']['all'], 4), 'data swap moved by step', moved)
    if not guards['no_draws_outside_the_acquired_region']:
        problems.append(f"{name}: up to {guards['most_draws_outside_the_acquired_region']} draws outside the acquired region")
    if scored['counterfactual_with_own_data_reproduces_every_step'] is not True:
        problems.append(f'{name}: the swap function does not reproduce its own steps')
    if name.startswith('Fb_'):
        cost = scored['swap_cost']['psnr']['all']
        if any(value != 0 for value in moved) or cost['mean'] != 0 or cost['ci95'] != [0.0, 0.0]:
            problems.append(f"{name}: the twin's swap null fails (moved {moved}, cost {cost['mean']} {cost['ci95']})")
    elif not any(value > 0 for value in moved):
        problems.append(f'{name}: a data swap moved nothing, so it reads no data')
print('\n'.join(problems) or 'twin nulls hold, the arms read their data, every arm reproduces its own steps '
      'and no draw lands outside the acquired region')
sys.exit(1 if problems else 0)
EOF
}

run_info_commits() {
    python - "$OUT" "$EXPECTED_COMMIT" "$((${#RESUME_ARMS[@]} + ${#SMOKE_ARMS[@]}))" <<'EOF'
import json, sys
from pathlib import Path

infos = {path.parent.name: json.loads(path.read_text()) for path in Path(sys.argv[1]).glob('*/run_info.json')}
bad = {name: (info['commit'], info['dirty_tracked_files']) for name, info in infos.items()
       if info['commit'] != sys.argv[2] or info['dirty_tracked_files']}
print(f'{len(infos)} runs recorded', {name: info['commit'] for name, info in infos.items()})
print(f'not the shipped commit, or dirty: {bad}' if bad else 'every run recorded the shipped commit, clean')
sys.exit(1 if bad or len(infos) != int(sys.argv[3]) else 0)
EOF
}

check commit shipped_commit
check tests python -m pytest -q tests
for arm in ${RESUME_ARMS[@]+"${RESUME_ARMS[@]}"}; do
    check "resume_$arm" resume "$arm"
    check "continues_$arm" resume_continues "$arm"
done
for arm in ${SMOKE_ARMS[@]+"${SMOKE_ARMS[@]}"}; do
    check "smoke_$arm" smoke "$arm"
    check "score_$arm" score "$arm"
done
[[ ${#SMOKE_ARMS[@]} -gt 0 ]] && check nulls nulls
check run_info run_info_commits

echo
failed=0
for name in "${CHECKS[@]}"; do
    printf '  %-40s %s\n' "$name" "${RESULT[$name]}"
    [[ "${RESULT[$name]}" == pass ]] || failed=1
done
echo "logs: $OUT"
exit "$failed"
