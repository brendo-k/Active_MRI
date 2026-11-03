"""Phase timing of one VarNet+LOUPE training step (R=8, 320^2, 12 cascades, bs 16) as Sequential_Trainer runs it.

Usage: CUDA_VISIBLE_DEVICES=N python profile_step.py <repo> [steps] [extra hydra overrides...]
Prints mean ms per phase over `steps` steps after warmup, with a sync between phases, plus the unsynced step time.
"""
import os
import sys
import time
import json

os.environ.setdefault('WANDB_MODE', 'disabled')
repo = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 12
extra = sys.argv[3:]
sys.path.insert(0, repo)
os.chdir(repo)

import torch
from hydra import compose, initialize_config_dir

BASE = ['recon=varnet', 'datasets=fastmri_local_v1_flair_knee', 'misc.resolution=[320,320]', 'sampler.accelerations=[8.0]',
        'sampler.budget_over=acquired', 'training.num_step=1', 'training.batch_size=16', 'training.num_epochs=40',
        'training.lr_step_size=36', 'training.seed=42', 'sampler.sampler_type=loupe_2d', 'sampler.loupe_vd_init_power=4',
        'sampler.anatomy_conditioning=false', 'datasets.limit_volumes=0.2', 'output.out_dir=/tmp/claude-2081/prof_out']

with initialize_config_dir(config_dir=os.path.join(repo, 'configs'), version_base=None):
    cfg = compose(config_name='config', overrides=BASE + extra)

from activemri.training.builders import build_dataloaders, build_train_config
from activemri.training.trainer import Sequential_Trainer, _apply_signal_mask
from activemri.utils.evaluate import build_metrics
from activemri.utils.metrics import MaskedMetricAccumulator
import train as train_module

config = build_train_config(cfg)
train_module.seed_everything(cfg.training.seed)
torch.backends.cudnn.benchmark = config.cudnn_benchmark
train_loader, val_loader = build_dataloaders(cfg)
trainer = Sequential_Trainer(config, train_loader, torch.device('cuda'), val_loader=val_loader)
model = trainer.model
model.train()

def sync():
    torch.cuda.synchronize()

phases = {k: [] for k in ['data', 'h2d', 'forward', 'loss', 'backward', 'grad_stats_clip', 'opt_step', 'metrics', 'total']}
metrics = build_metrics(device='cuda')
masked = MaskedMetricAccumulator(device='cuda')
it = iter(train_loader)
n_warm = 4
for step in range(n_warm + steps):
    t0 = time.perf_counter()
    try:
        batch = next(it)
    except StopIteration:
        it = iter(train_loader)
        batch = next(it)
    t1 = time.perf_counter(); sync()
    kspace = trainer._load_kspace(batch)
    target = trainer._target(batch, kspace)
    sync(); t2 = time.perf_counter()
    pred = trainer._train_forward(target, kspace, batch)
    sync(); t3 = time.perf_counter()
    target_dict = trainer._target_dict(batch, target, kspace)
    meta = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0}
    loss, _ = model.loss(pred, target_dict, meta, config.loss_type)
    sync(); t4 = time.perf_counter()
    trainer.optimizer.zero_grad()
    loss.backward()
    sync(); t5 = time.perf_counter()
    from activemri.training.trainer import gradient_stats, clip_gradients_per_group, zero_coil_fraction
    grad_log = gradient_stats(model, config.grad_clip_norm)
    clip_gradients_per_group(model, config.grad_clip_norm)
    zero_coil_fraction(kspace)
    sync(); t6 = time.perf_counter()
    trainer.optimizer.step()
    sync(); t7 = time.perf_counter()
    output = pred['output']
    tm = target.detach()
    if output.dim() != tm.dim():
        tm = tm.unsqueeze(1)
    metrics['PSNR'].data_range = tm.max(); metrics['SSIM'].data_range = tm.max()
    signal_mask = trainer._metric_mask(batch)
    mo, mt = _apply_signal_mask(output.detach(), tm, signal_mask)
    metrics.update(mo, mt)
    masked.update(output.detach(), tm, signal_mask)
    loss.item()
    sync(); t8 = time.perf_counter()
    if step >= n_warm:
        for k, v in zip(['data', 'h2d', 'forward', 'loss', 'backward', 'grad_stats_clip', 'opt_step', 'metrics', 'total'],
                        [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t8 - t7, t8 - t0]):
            phases[k].append(1000 * v)
    print(f"step {step} total {1000*(t8-t0):.0f} ms fwd {1000*(t3-t2):.0f} bwd {1000*(t5-t4):.0f} data {1000*(t1-t0):.0f} "
          f"metrics {1000*(t8-t7):.0f} peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)

summary = {k: sum(v) / len(v) for k, v in phases.items()}
print(json.dumps({'phases_ms': summary, 'peak_GiB': torch.cuda.max_memory_allocated() / 2**30, 'extra': extra}, indent=1))

# Unsynced: how the trainer actually runs (no sync between phases)
t0 = time.perf_counter()
n = 0
for step in range(steps):
    try:
        batch = next(it)
    except StopIteration:
        it = iter(train_loader)
        batch = next(it)
    kspace = trainer._load_kspace(batch)
    target = trainer._target(batch, kspace)
    pred = trainer._train_forward(target, kspace, batch)
    target_dict = trainer._target_dict(batch, target, kspace)
    loss, _ = model.loss(pred, target_dict, meta, config.loss_type)
    trainer.optimizer.zero_grad()
    loss.backward()
    clip_gradients_per_group(model, config.grad_clip_norm)
    trainer.optimizer.step()
    n += 1
sync()
print(json.dumps({'unsynced_bare_step_ms': 1000 * (time.perf_counter() - t0) / n}))
