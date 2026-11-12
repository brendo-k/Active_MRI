"""The CoreHPC launcher (scripts/launch/hpc/multistep_adaptivity/common.sh) builds valid train.py arguments for every arm.

Run with ``.venv/bin/python -m pytest tests/test_multistep_launch.py`` from the repo root.

Only CoreHPC builds these arguments in full: a lab-server dry run (LAB_SERVER_TEST=1) leaves the data roots out. So each
arm's argument list is built here as train.sbatch builds it on CoreHPC, then composed through train.py's own override
path and Hydra. A quoting error, an unknown key or a wrong recipe fails here instead of after a queue wait (validation
job 2996412 lost every check to an unquoted ~ in a data root).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import train  # noqa: E402

LAUNCH = REPO / 'scripts' / 'launch' / 'hpc' / 'multistep_adaptivity'
COREHPC_ROOTS = {'hpc': '~/scratch/data/fastmri/{anatomy}/data', 'local_batch2': '~/scratch/data/fastmri/knee/data'}
ARMS = (['B_42', 'B_43']
        + [f'{kind}_{steps}_s{seed}' for seed in (42, 43) for steps in (1, 4, 16) for kind in 'AT']
        + ['T_1_s42_f142', 'T_16_s42_f142', 'A_4_s42_rec', 'A_16_s42_img', 'A_16_s42_scr', 'T_16_s42_scr',
           'A_16_s42_kscr1', 'T_16_s42_kscr1', 'T_16_s142_kscr1'])


def corehpc_arguments(arm, exp_dir):
    """train.sbatch's `train.py "${ARM_ARGS[@]}" "${RECIPE[@]}" output.out_dir=...`, as common.sh builds it on CoreHPC."""
    env = {key: value for key, value in os.environ.items() if key != 'LAB_SERVER_TEST'}
    env.update(GPU_NAME='test', EXP_DIR=str(exp_dir))
    script = f'source "{LAUNCH}/common.sh" && arm_args "$1" && printf "%s\\0" "${{ARM_ARGS[@]}}" "${{RECIPE[@]}}"'
    built = subprocess.run(['bash', '-c', script, 'bash', arm], env=env, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    return built.stdout.split('\0')[:-1] + [f'output.out_dir={exp_dir}/runs/{arm}']


def composed(arguments):
    parser = train.build_parser()
    args, raw_overrides = parser.parse_known_args(arguments)
    overrides = [o for o in train.build_overrides(parser, args, raw_overrides) if not o.startswith('hydra.')]
    with initialize_config_dir(version_base=None, config_dir=str(REPO / 'configs')):
        return compose(config_name='config', overrides=overrides)


@pytest.fixture
def exp_dir(tmp_path):
    for seed in (42, 43):
        base = tmp_path / 'runs' / f'B_{seed}' / 'last_model.pt'
        base.parent.mkdir(parents=True)
        base.touch()
    return tmp_path


@pytest.mark.parametrize('arm', ARMS)
def test_every_arm_composes_with_the_corehpc_data_roots_and_its_recipe(arm, exp_dir):
    cfg = composed(corehpc_arguments(arm, exp_dir))
    assert dict(cfg.datasets.roots) == COREHPC_ROOTS
    base_seed = re.search(r'(?:^B_|_s)(\d+)', arm).group(1)
    floor_seed = re.search(r'_f(\d+)', arm)
    assert cfg.training.seed == int(floor_seed.group(1) if floor_seed else base_seed)
    if arm.startswith('B_'):
        assert (cfg.sampler.sampler_type, cfg.training.num_step, cfg.training.num_epochs) == ('loupe_2d', 1, 100)
        return
    assert cfg.sampler.sampler_type == 'seq_adapt_2d'
    assert cfg.training.num_step == int(arm.split('_')[1])
    assert cfg.sampler.data_blind == arm.startswith('T_')
    assert cfg.sampler.input == ('recon' if arm.endswith('_rec') else 'measured')
    assert list(cfg.sampler.data_channels) == (['image'] if arm.endswith('_img') else ['image', 'kspace'])
    knee_subproblem = '_kscr' in arm
    from_scratch = arm.endswith('_scr') or knee_subproblem
    base = None if from_scratch else str(exp_dir / 'runs' / f'B_{base_seed}' / 'last_model.pt')
    assert (cfg.output.pretrained_recon, cfg.sampler.init_mask_checkpoint) == (base, base)
    assert cfg.training.num_epochs == (100 if from_scratch else 30)
    assert list(cfg.datasets.classes) == (['knee'] if knee_subproblem else ['FLAIR', 'knee'])
    assert cfg.datasets.val_half == ('dev' if knee_subproblem else 'all')
    assert cfg.sampler.loupe_vd_init_power == (0 if knee_subproblem else 4 if from_scratch else cfg.sampler.loupe_vd_init_power)


def test_an_undefined_knee_subproblem_iteration_is_refused(exp_dir):
    with pytest.raises(AssertionError, match='iteration 9 is not defined'):
        corehpc_arguments('A_16_s42_kscr9', exp_dir)
