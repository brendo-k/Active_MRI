#!/bin/bash
# The CoreHPC validation job of the multistep_adaptivity proposal (P0 item 7). Every check below must pass on the pinned
# GPU type before any P1 run is submitted. Each check writes $EXP_DIR/validate/<job id>/<check>.log, and the job ends
# with a pass/FAIL table. Submit from $EXP_DIR (see run.md):
#
#   sbatch --gres=gpu:<pinned type>:1 --export=ALL,GPU_NAME=<pinned name>,EXPECTED_COMMIT=<shipped hash> \
#       code/scripts/launch/hpc/multistep_adaptivity/validate.sbatch
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
    local name="$1" steps="$2" blind="$3"
    shift 3
    python train.py --model sequential "${RECIPE[@]}" --seed 42 "output.out_dir=$OUT/$name" \
        sampler.sampler_type=seq_adapt_2d sampler.anatomy_conditioning=true sampler.loupe_vd_init_power=4 \
        training.num_step="$steps" sampler.data_blind="$blind" training.num_epochs=1 training.limit_batches=0.1 \
        training.lr=1e-4 training.lr_scheduler=constant training.train_metrics_every=1 "$@"
}

recon_input_smoke() {
    local base="$OUT/smoke_base/last_model.pt"
    smoke smoke_rec 4 false sampler.loupe_vd_init_power=0 "output.pretrained_recon=$base" \
        "sampler.init_mask_checkpoint=$base" sampler.input=recon "sampler.input_recon_checkpoint=$base"
}

base_smoke() {
    python train.py --model loupe "${RECIPE[@]}" --seed 42 "output.out_dir=$OUT/$1" training.num_step=1 \
        training.num_epochs=1 training.limit_batches=0.1 training.lr_step_size=90 sampler.loupe_vd_init_power=4 \
        sampler.anatomy_conditioning=false
}

score() {
    python scripts/eval/eval_multistep_adaptivity.py score --run "$OUT/$1" --name "$1" --out "$OUT/scores" --limit 120 \
        --no-wandb "${@:2}"
}

twin_nulls() {
    python - "$OUT/scores" <<'EOF'
import json, sys
from pathlib import Path

scores = Path(sys.argv[1])
twin, adaptive = (json.loads((scores / f'{name}.json').read_text()) for name in ('smoke_T4', 'smoke_A4'))
problems = []
if twin['swap_null']['donor_final_mask_equals_own'] != 1.0:
    problems.append(f"twin: donor's final mask equals its own in {twin['swap_null']['donor_final_mask_equals_own']} of slices")
if any(moved != 0 for moved in twin['data_swap']['moved_by_step']['all']):
    problems.append(f"twin: a data swap moved samples {twin['data_swap']['moved_by_step']['all']}")
cost = twin['swap_cost']['psnr']['all']
if cost['mean'] != 0 or cost['ci95'] != [0.0, 0.0]:
    problems.append(f"twin: swap cost {cost['mean']} {cost['ci95']}")
for name, scored in (('twin', twin), ('adaptive', adaptive)):
    if scored['counterfactual_with_own_data_reproduces_every_step'] is not True:
        problems.append(f'{name}: the swap function does not reproduce its own steps')
if not any(moved > 0 for moved in adaptive['data_swap']['moved_by_step']['all']):
    problems.append('adaptive: a data swap moved nothing, so the positive control failed')
print('twin swap null', twin['swap_null'], 'twin moved', twin['data_swap']['moved_by_step']['all'], 'twin cost', cost)
print('adaptive moved', adaptive['data_swap']['moved_by_step']['all'], 'adaptive mask equality',
      adaptive['swap_null']['donor_final_mask_equals_own'])
print('\n'.join(problems) or 'nulls hold, and the positive control is above 0')
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
sys.exit(1 if bad or not infos else 0)
EOF
}

check commit shipped_commit
check tests python -m pytest -q tests
check data python scripts/diagnostics/multistep_data_check.py \
    --check scripts/launch/hpc/multistep_adaptivity/reference_slices.json "${DATA_OVERRIDES[@]}"
check smoke_A4 smoke smoke_A4 4 false
check smoke_T4 smoke smoke_T4 4 true
check score_A4 score smoke_A4
check score_T4 score smoke_T4
check nulls twin_nulls
check smoke_A16 smoke smoke_A16 16 false
check smoke_base base_smoke smoke_base
check score_base score smoke_base --no-sensitivity
check smoke_rec recon_input_smoke
check score_rec score smoke_rec
check smoke_img smoke smoke_img 4 false 'sampler.data_channels=[image]'
check run_info run_info_commits

echo
echo "Epoch lines (10% of a train epoch, compile included, then the full val set):"
for name in smoke_base smoke_A4 smoke_T4 smoke_A16 smoke_rec smoke_img; do
    echo "  $name: $(grep -h 'Epoch = ' "$OUT/$name.log" | tail -n 1 | sed 's/.*\(TrainTime.*\)/\1/')"
done
echo
failed=0
for name in "${CHECKS[@]}"; do
    printf '  %-11s %s\n' "$name" "${RESULT[$name]}"
    [[ "${RESULT[$name]}" == pass ]] || failed=1
done
echo "logs: $OUT"
exit "$failed"
