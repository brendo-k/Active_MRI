#!/bin/bash
# The loss_domain__loupe_loss_partial_fourier validation job (reports/loss_domain__loupe_loss_partial_fourier/
# proposal.md). Training jobs are submitted with --dependency=afterok on it, so they start only if every check passes:
# - the shipped commit, clean, and the test suite;
# - each arm kind trains for 10% of one epoch exactly as train.sbatch does (seed 42) and is scored with --limit 120:
#   a finite loss, realised fraction 0.125 +/- 0.002, no draw outside the acquired region;
# - the symmetric arms' learned maps are exactly point-symmetric and the free arms' are not;
# - every run recorded the shipped commit.
# Submit from $EXP_DIR (~/scratch/loss_domain__loupe_loss_partial_fourier):
#
#   sbatch --gres=gpu:nvidia_rtx_pro_6000:1 --export=ALL,GPU_NAME='RTX PRO 6000',EXPECTED_COMMIT=<shipped hash> \
#       code/scripts/launch/hpc/loss_domain__loupe_loss_partial_fourier/validate.sbatch
#
#SBATCH --job-name=loupe_lpf_validate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=04:00:00
#SBATCH --output=slurm_log/%x-%j.log

set -uo pipefail
TOPIC=loss_domain__loupe_loss_partial_fourier
source "${EXP_DIR:-$HOME/scratch/$TOPIC}/code/scripts/launch/hpc/$TOPIC/common.sh"
: "${EXPECTED_COMMIT:?submit with --export=ALL,GPU_NAME=<card>,EXPECTED_COMMIT=<hash>}"
start_job
OUT="$EXP_DIR/validate/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
ARMS=(Lrss_1_s42_tF Lcplx_1_s42_tF Lrsssym_1_s42_tF Lcplxsym_1_s42_tF)
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
    (loss_pf_arm_args "$1" && python train.py "${ARM_ARGS[@]}" "${RECIPE[@]}" "output.out_dir=$OUT/$1" \
        training.num_epochs=1 training.limit_batches=0.1)
}

score() {
    python scripts/eval/eval_multistep_adaptivity.py score --run "$OUT/$1" --name "$1" --out "$OUT/scores" --limit 120 \
        --no-wandb --gradients
}

arms_hold() {
    python - "$OUT" "${ARMS[@]}" <<'EOF'
import json, re, sys
from pathlib import Path

import torch

from activemri.samplers.loupe import point_reflected

out, problems = Path(sys.argv[1]), []
for name in sys.argv[2:]:
    scored = json.loads((out / 'scores' / f'{name}.json').read_text())
    realised = scored['means']['realised_fraction']['all']
    losses = re.findall(r'TrainLoss = ([-\d.eE+naif]+)', (out / f'smoke_{name}.log').read_text())
    checkpoint = torch.load(out / name / 'last_model.pt', map_location='cpu', weights_only=False)
    logits = checkpoint['model']['sampler.mask_net.mask']
    effective = (logits + point_reflected(logits)) / 2 if checkpoint['options'].loupe_point_symmetric else logits
    symmetric = torch.equal(effective, point_reflected(effective))
    print(f'{name}: realised {realised:.4f}, train loss {losses[-1] if losses else None}, loss type '
          f"{checkpoint['options'].loss_type}, effective map point-symmetric {symmetric}")
    if abs(realised - 0.125) > 0.002 or not scored['guards']['no_draws_outside_the_acquired_region']:
        problems.append(f'{name}: realised {realised} or draws outside the acquired region')
    if not losses or not torch.isfinite(torch.tensor(float(losses[-1]))):
        problems.append(f'{name}: no finite training loss')
    if symmetric != ('sym' in name):
        problems.append(f"{name}: effective map point-symmetric is {symmetric}")
    if (checkpoint['options'].loss_type == 'complex_l1') != ('cplx' in name):
        problems.append(f"{name}: trained with loss_type {checkpoint['options'].loss_type}")
print('\n'.join(problems) or 'every arm trains with its loss, keeps the budget, and only the symmetric maps are symmetric')
sys.exit(1 if problems else 0)
EOF
}

run_info_commits() {
    python - "$OUT" "$EXPECTED_COMMIT" "${#ARMS[@]}" <<'EOF'
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
for arm in "${ARMS[@]}"; do
    check "smoke_$arm" smoke "$arm"
    check "score_$arm" score "$arm"
done
check arms arms_hold
check run_info run_info_commits

echo
failed=0
for name in "${CHECKS[@]}"; do
    printf '  %-28s %s\n' "$name" "${RESULT[$name]}"
    [[ "${RESULT[$name]}" == pass ]] || failed=1
done
echo "logs: $OUT"
exit "$failed"
