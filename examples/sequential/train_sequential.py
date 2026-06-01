# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import random

import numpy as np
import torch
import os
import sys

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import activemri.envs.loupe_envs as loupe_envs
from activemri.baselines.sequentail_trainer import Sequential_Trainer, NonRLTester
from activemri.config import TrainConfig
import matplotlib
matplotlib.use('Agg')


def build_arg_parser():
    parser = argparse.ArgumentParser(description='MRI Reconstruction Example')
    parser.add_argument('--num-pools', type=int, default=4)
    parser.add_argument('--num-step', type=int, default=1)
    parser.add_argument('--drop-prob', type=float, default=0.0)
    parser.add_argument('--num-chans', type=int, default=64)

    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--noise-type', type=str, default='none')
    parser.add_argument('--noise-level', type=float, default=5e-5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-step-size', type=int, default=40)
    parser.add_argument('--lr-gamma', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.)
    parser.add_argument('--report-interval', type=int, default=100)
    parser.add_argument('--data-parallel', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exp-dir', type=pathlib.Path, required=True)
    parser.add_argument('--checkpoint1', type=str)
    parser.add_argument('--entropy_weight', type=float, default=0.0)
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--sparsity_weight', type=float, default=0.0)
    parser.add_argument('--save-model', type=bool, default=False)

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resolution', default=[128, 128], nargs='+', type=int)

    parser.add_argument('--data-path', type=pathlib.Path, required=False)
    parser.add_argument('--dataset-name', type=str,
                        choices=['fashion-mnist', 'dicom-knee', 'real-knee', 'brain'], required=True)
    parser.add_argument('--sample-rate', type=float, default=1.)

    parser.add_argument('--accelerations', nargs='+', default=[4], type=float)
    parser.add_argument('--label_range', nargs='+', type=int, default=None)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input_chans', type=int, choices=[1, 2], required=True)
    parser.add_argument('--output_chans', type=int, default=2)
    parser.add_argument('--line-constrained', action='store_true')
    parser.add_argument('--unet', action='store_true')
    parser.add_argument('--preselect', action='store_true', default=True)
    parser.add_argument('--conjugate_mask', action='store_true')
    parser.add_argument('--loss_type', type=str, choices=['l1', 'ssim', 'nmse', 'psnr', 'xentropy'], default='l1')
    parser.add_argument('--test_visual_frequency', type=int, default=1000)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--bi-dir', action='store_true')
    parser.add_argument('--preselect_num', type=int, default=10)
    parser.add_argument('--binary_sampler', action='store_true')
    parser.add_argument('--clamp', type=float, default=1e10)
    parser.add_argument('--old_recon', action='store_true')
    parser.add_argument('--uncertainty_loss', action='store_true')
    parser.add_argument('--uncertainty_weight', type=float, default=0)
    parser.add_argument('--detach_kspace', action='store_true')
    parser.add_argument('--random_rotate', action='store_true')
    parser.add_argument('--kspace_weight', type=float, default=0)
    parser.add_argument('--pretrained_recon', action='store_true')
    parser.add_argument('--limit_batches', type=float, default=1.0)
    parser.add_argument('--recon_type', type=str, choices=['unet', 'lls'], default='unet')
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--cg_lamda', type=float, default=0.0)
    parser.add_argument('--sense_tikhonov_weight', type=float, default=0.0)

    return parser.parse_args()


def config_from_args(args) -> TrainConfig:
    config = TrainConfig(
        num_pools=args.num_pools,
        num_step=args.num_step,
        drop_prob=args.drop_prob,
        num_chans=args.num_chans,
        model=args.model,
        input_chans=args.input_chans,
        output_chans=args.output_chans,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        report_interval=args.report_interval,
        limit_batches=args.limit_batches,
        loss_type=args.loss_type,
        recon_weight=args.recon_weight,
        entropy_weight=args.entropy_weight,
        sparsity_weight=args.sparsity_weight,
        kspace_weight=args.kspace_weight,
        uncertainty_loss=args.uncertainty_loss,
        uncertainty_weight=args.uncertainty_weight,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        device=args.device,
        data_parallel=args.data_parallel,
        exp_dir=args.exp_dir,
        checkpoint1=args.checkpoint1,
        resume=args.checkpoint1 is not None,
        dataset_name=args.dataset_name,
        data_path=str(args.data_path) if args.data_path else None,
        sample_rate=args.sample_rate,
        resolution=args.resolution,
        label_range=args.label_range,
        seed=args.seed,
        accelerations=args.accelerations,
        line_constrained=args.line_constrained,
        conjugate_mask=args.conjugate_mask,
        preselect=args.preselect,
        preselect_num=args.preselect_num,
        bi_dir=args.bi_dir,
        binary_sampler=args.binary_sampler,
        clamp=args.clamp,
        unet=args.unet,
        old_recon=args.old_recon,
        detach_kspace=args.detach_kspace,
        pretrained_recon=args.pretrained_recon,
        recon_type=args.recon_type,
        cg_iters=args.cg_iters,
        cg_lamda=args.cg_lamda,
        sense_tikhonov_weight=args.sense_tikhonov_weight,
        random_rotate=args.random_rotate,
        save_model=args.save_model,
        test=args.test,
        test_visual_frequency=args.test_visual_frequency,
    )

    if config.num_step == 0:
        config.fixed_input = True
        config.num_step = 1

    if config.test:
        config.batch_size = 1

    return config


def resolve_exp_dir(config: TrainConfig) -> None:
    if str(config.exp_dir) == 'auto':
        noise_str = '_no_noise_' if config.noise_type == 'none' else f'_{config.noise_type}{config.noise_level}_'
        base_name = (
            f'checkpoints/{config.dataset_name}_{float(config.accelerations[0])}x_{config.model}'
            f'_bi_dir_{config.bi_dir}_step_{config.num_step}'
            f'_preselect_{config.preselect}{noise_str}lr={config.lr}'
            f'_bs={config.batch_size}_loss_type={config.loss_type}_epochs={config.num_epochs}'
        )
        existing = (
            sorted(pathlib.Path('checkpoints').glob(pathlib.Path(base_name).name + '_run_*'))
            if pathlib.Path('checkpoints').exists() else []
        )
        run_num = len(existing) + 1
        config.exp_dir = pathlib.Path(base_name + f'_run_{run_num:03d}')

    config.visualization_dir = config.exp_dir / 'visualizations'


def build_env(config: TrainConfig):
    if config.dataset_name == 'fashion-mnist':
        config.data_path = 'datasets/fashion-mnist'
        return loupe_envs.LOUPEFashionMNISTEnv(config)
    elif config.dataset_name == 'dicom-knee':
        config.data_path = 'datasets/knee'
        return loupe_envs.LOUPEActiveMRIEnv(config)
    elif config.dataset_name == 'brain':
        if config.data_path is None:
            config.data_path = 'datasets/brain'
        return loupe_envs.LoupeBrainEnv(config)
    elif config.dataset_name == 'real-knee':
        config.data_path = 'datasets/knee'
        return loupe_envs.LOUPERealKspaceEnv(config)
    elif config.dataset_name == 'synthetic':
        config.data_path = 'datasets/synthetic'
        return loupe_envs.LOUPESyntheticEnv(config)
    else:
        raise NotImplementedError(f'Unknown dataset: {config.dataset_name}')


if __name__ == "__main__":
    args = build_arg_parser()
    config = config_from_args(args)
    resolve_exp_dir(config)

    print(f'save logs to {config.exp_dir}')

    env = build_env(config)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.test:
        policy = NonRLTester(env, config.exp_dir, config, None)
    else:
        policy = Sequential_Trainer (config, env, torch.device(config.device))
    policy()
