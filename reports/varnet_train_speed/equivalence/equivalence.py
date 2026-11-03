"""Train a few steps with given trainer overrides and save per-parameter checksums + RNG state, to check that the
throughput switches (train_metrics_every, visualize_every, val_every) leave the training math unchanged.

Usage: CUDA_VISIBLE_DEVICES=N python equivalence.py <repo> <out.pt> [hydra overrides...]
Deterministic: 2 cascades, no compile, cuDNN autotuner off, deterministic algorithms where available.
"""
import os
import sys

os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
repo, out = sys.argv[1], sys.argv[2]
extra = sys.argv[3:]
sys.path.insert(0, repo)
os.chdir(repo)

import torch
from hydra import compose, initialize_config_dir

torch.use_deterministic_algorithms(True, warn_only=True)
BASE = ['recon=varnet', 'datasets=fastmri_local_v1_flair_knee', 'misc.resolution=[320,320]', 'sampler.accelerations=[8.0]',
        'sampler.budget_over=acquired', 'training.num_step=1', 'training.batch_size=8', 'training.num_epochs=2',
        'training.lr_step_size=36', 'training.seed=42', 'sampler.sampler_type=loupe_2d', 'sampler.loupe_vd_init_power=4',
        'sampler.anatomy_conditioning=false', 'datasets.limit_volumes=0.2', 'recon.num_cascades=2',
        'training.compile_unets=false', 'training.cudnn_benchmark=false', 'training.limit_batches=0.1',
        f'output.out_dir={os.path.dirname(out)}/equiv_run_{os.path.basename(out)}']
with initialize_config_dir(config_dir=os.path.join(repo, 'configs'), version_base=None):
    cfg = compose(config_name='config', overrides=BASE + extra)

from activemri.training.builders import build_dataloaders, build_train_config
from activemri.training.trainer import Sequential_Trainer
import train as train_module

config = build_train_config(cfg)
train_module.seed_everything(cfg.training.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
train_loader, val_loader = build_dataloaders(cfg)
trainer = Sequential_Trainer(config, train_loader, torch.device('cuda'), val_loader=val_loader)
losses = []
for epoch in range(2):
    trainer.epoch = epoch
    trainer.scheduler.step(epoch)
    loss, _, _ = trainer.train_epoch()
    losses.append(loss)
state = {name: p.detach().double().sum().item() for name, p in trainer.model.named_parameters()}
torch.save({'losses': losses, 'param_sums': state, 'cuda_rng': torch.cuda.get_rng_state(), 'cpu_rng': torch.get_rng_state(),
            'params': {n: p.detach().cpu() for n, p in trainer.model.named_parameters()}, 'extra': extra}, out)
print('losses', losses)
