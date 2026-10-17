"""Single entry point for training one sampler of each kind.

    python train.py --model loupe     --recon varnet --dataset brain_knee_balanced --acceleration 8
    python train.py --model conv      --recon unet
    python train.py --model mnet      --recon varnet --acceleration 8 --pretrained-recon <random-mask recon checkpoint>
    python train.py --model reinforce --recon varnet --entropy-weight 0.005
    python train.py --model ppo       --recon varnet --acceleration 8
    python train.py --model dqn       --recon varnet --acceleration 4

loupe, conv and reinforce train through Sequential_Trainer (configs/config.yaml); mnet through its subclass
MNetTrainer, which alternates mask refinement and sampler fitting (activemri/training/mnet_trainer.py). `sequential` is the
same trainer with the sampler left to sampler.sampler_type (caipi_2d, autosamp_2d, the 1D samplers). ppo and dqn
train through their own episode loops (configs/train_ppo.yaml, configs/train_dqn.yaml). Every flag
left unset keeps that model's YAML value. Trailing key=value arguments are Hydra overrides that are
applied last, so they win over the presets and flags:

    python train.py --model loupe --recon varnet training.num_epochs=30 sampler.slope=5
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
for _pool in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_pool, '8')

import argparse
import random
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

MODELS = ('loupe', 'conv', 'mnet', 'reinforce', 'ppo', 'dqn', 'sequential')
SEQUENTIAL_MODELS = ('loupe', 'conv', 'mnet', 'reinforce', 'sequential')
RL_MODELS = ('ppo', 'dqn')

JOB_NAME = {model: 'train_sequential' for model in SEQUENTIAL_MODELS} | {'ppo': 'train_ppo', 'dqn': 'train_dqn'}

PRESET_OVERRIDES = {
    'loupe': {'sampler.sampler_type': 'loupe_2d'},
    'conv': {'sampler.sampler_type': 'conv_2d'},
    'mnet': {'sampler.sampler_type': 'mnet_2d', 'training.batch_size': 8},
    'reinforce': {'sampler.sampler_type': 'loupe_2d', 'sampler.mask_train_mode': 'reinforce'},
    'sequential': {},
    'ppo': {},
    'dqn': {},
}

# flag -> config key, per trainer family. A flag a family has no key for is rejected.
FLAG_KEYS = {
    'recon': {'sequential': 'recon', 'ppo': 'recon', 'dqn': 'recon'},
    'dataset': {'sequential': 'datasets', 'ppo': 'datasets', 'dqn': 'datasets'},
    'batch_size': {'sequential': 'training.batch_size', 'ppo': 'training.batch_size', 'dqn': 'training.batch_size'},
    'seed': {'sequential': 'training.seed', 'ppo': 'training.seed', 'dqn': 'training.seed'},
    'out_dir': {'sequential': 'output.out_dir', 'ppo': 'output.out_dir', 'dqn': 'output.out_dir'},
    'lr': {'sequential': 'training.lr', 'ppo': 'ppo.learning_rate', 'dqn': 'dqn.dqn_learning_rate'},
    'acceleration': {'sequential': 'sampler.accelerations', 'ppo': 'env.acceleration', 'dqn': 'env.acceleration'},
    'epochs': {'sequential': 'training.num_epochs'},
    'pretrained_recon': {'sequential': 'output.pretrained_recon', 'ppo': 'ppo.pretrained_checkpoint',
                         'dqn': 'dqn.pretrained_checkpoint'},
    'entropy_weight': {'sequential': 'sampler.reinforce_entropy_weight'},
}


def family(model: str) -> str:
    return 'sequential' if model in SEQUENTIAL_MODELS else model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Unset flags keep the model YAML default. Trailing key=value arguments are Hydra overrides.')
    parser.add_argument('--model', required=True, choices=MODELS, help='which sampler to train')
    parser.add_argument('--recon', choices=['unet', 'varnet', 'cg_sense'],
                        help='reconstructor config group (ppo and dqn require varnet)')
    parser.add_argument('--dataset', help='datasets config group, e.g. brain, knee, brain_knee_balanced')
    parser.add_argument('--acceleration', type=float, help='acceleration factor R')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int, help='loupe/conv/reinforce only; ppo and dqn run for steps, set via overrides')
    parser.add_argument('--lr', type=float, help='sampler/reconstructor lr (ppo: policy lr, dqn: q-network lr)')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--out-dir', help='run directory for checkpoints, logs and the resolved config')
    parser.add_argument('--pretrained-recon', help='reconstructor checkpoint to warm-start from '
                        '(ppo/dqn: the frozen env reconstructor)')
    parser.add_argument('--entropy-weight', type=float, help='reinforce only: mask-distribution entropy bonus')
    parser.add_argument('--print-config', action='store_true', help='print the resolved config and exit')
    return parser


def override_key(override: str) -> str:
    return override.lstrip('+~').split('=', 1)[0]


def build_overrides(parser: argparse.ArgumentParser, args: argparse.Namespace, raw_overrides: list[str]) -> list[str]:
    """Preset, then flags, then raw overrides; a later entry for the same key replaces the earlier one."""
    fam = family(args.model)
    if args.model in RL_MODELS and args.recon not in (None, 'varnet'):
        parser.error(f'--model {args.model} reads recon.num_cascades/pools from a VarNet; got --recon {args.recon}')

    merged = {key: f'{key}={value}' for key, value in PRESET_OVERRIDES[args.model].items()}
    merged['hydra.job.name'] = f'hydra.job.name={JOB_NAME[args.model]}'
    for flag, keys in FLAG_KEYS.items():
        value = getattr(args, flag)
        if value is None:
            continue
        if fam not in keys:
            parser.error(f'--{flag.replace("_", "-")} does not apply to --model {args.model}')
        key = keys[fam]
        merged[key] = f'{key}=[{value}]' if key == 'sampler.accelerations' else f'{key}={value}'

    for override in raw_overrides:
        if '=' not in override:
            parser.error(f'unrecognized argument {override!r} (overrides look like key=value)')
        merged[override_key(override)] = override

    return list(merged.values()) + (['--cfg', 'job'] if args.print_config else [])


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(version_base=None, config_path='configs', config_name='config')
def run_sequential(cfg: DictConfig):
    from activemri.training.builders import build_dataloaders, build_train_config
    from activemri.training.mnet_trainer import MNetTrainer
    from activemri.training.trainer import Sequential_Trainer

    print(OmegaConf.to_yaml(cfg))
    config = build_train_config(cfg)
    seed_everything(cfg.training.seed)
    train_loader, val_loader = build_dataloaders(cfg)
    trainer_class = MNetTrainer if config.sampler_type == 'mnet_2d' else Sequential_Trainer
    trainer_class(config, train_loader, torch.device(config.device), val_loader=val_loader)()


@hydra.main(version_base=None, config_path='configs', config_name='train_ppo')
def run_ppo(cfg: DictConfig):
    from activemri.training.ppo_trainer import PPOTrainer
    from activemri.training.rl_builders import build_ppo_config, build_ppo_envs

    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.training.seed)
    train_env, val_env = build_ppo_envs(cfg)
    PPOTrainer(build_ppo_config(cfg), train_env, val_env, torch.device(cfg.device))()


@hydra.main(version_base=None, config_path='configs', config_name='train_dqn')
def run_dqn(cfg: DictConfig):
    from activemri.training.dqn_trainer import DQNTrainer
    from activemri.training.rl_builders import build_dqn_config, build_dqn_envs

    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.training.seed)
    train_env, val_env = build_dqn_envs(cfg)
    DQNTrainer(build_dqn_config(cfg), train_env, val_env, torch.device(cfg.device))()


RUNNERS = {'sequential': run_sequential, 'ppo': run_ppo, 'dqn': run_dqn}


def main():
    parser = build_parser()
    args, raw_overrides = parser.parse_known_args()
    overrides = build_overrides(parser, args, raw_overrides)
    sys.argv = [sys.argv[0], *overrides]
    RUNNERS[family(args.model)]()


if __name__ == '__main__':
    main()
