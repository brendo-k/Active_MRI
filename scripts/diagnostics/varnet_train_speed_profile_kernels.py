"""CUDA kernel-time breakdown of the R=8 VarNet+LOUPE train step (forward, backward, optimizer; no trainer metrics),
grouped by kernel kind, at bs 16 / 12 cascades / 320x320 on 20% of the local v1 FLAIR+knee data (reports/varnet_train_speed).

    CUDA_VISIBLE_DEVICES=N python scripts/diagnostics/varnet_train_speed_profile_kernels.py <repo> <out.json> [hydra overrides...]
"""
import json
import os
import re
import sys
from collections import defaultdict

os.environ.setdefault('WANDB_MODE', 'disabled')
repo, out = sys.argv[1], sys.argv[2]
extra = sys.argv[3:]
sys.path.insert(0, repo)
os.chdir(repo)

import torch
from hydra import compose, initialize_config_dir
from torch.profiler import ProfilerActivity, profile, record_function

BASE = ['recon=varnet', 'datasets=fastmri_local_v1_flair_knee', 'misc.resolution=[320,320]', 'sampler.accelerations=[8.0]',
        'sampler.budget_over=acquired', 'training.num_step=1', 'training.batch_size=16', 'training.num_epochs=40',
        'training.lr_step_size=36', 'training.seed=42', 'sampler.sampler_type=loupe_2d', 'sampler.loupe_vd_init_power=4',
        'sampler.anatomy_conditioning=false', 'datasets.limit_volumes=0.2', 'recon.fused_coil_ops=false',
        'sampler.sense_mask_method=pre_network',
        'output.out_dir=/tmp/claude-2081/prof_kernels']
with initialize_config_dir(config_dir=os.path.join(repo, 'configs'), version_base=None):
    cfg = compose(config_name='config', overrides=BASE + extra)

from activemri.training.builders import build_dataloaders, build_train_config
from activemri.training.trainer import Sequential_Trainer, clip_gradients_per_group
import train as train_module

config = build_train_config(cfg)
train_module.seed_everything(cfg.training.seed)
torch.backends.cudnn.benchmark = config.cudnn_benchmark
train_loader, val_loader = build_dataloaders(cfg)
trainer = Sequential_Trainer(config, train_loader, torch.device('cuda'), val_loader=val_loader)
model = trainer.model
model.train()
getattr(model, 'store_visuals', None) is not None and setattr(model, 'store_visuals', False)
meta = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0}
batches = []
for i, batch in enumerate(train_loader):
    batches.append(batch)
    if len(batches) == 12:
        break


def step(batch):
    kspace = trainer._load_kspace(batch)
    target = trainer._target(batch, kspace)
    with record_function('forward'):
        pred = trainer._train_forward(target, kspace, batch)
        loss, _ = model.loss(pred, trainer._target_dict(batch, target, kspace), meta, config.loss_type)
    trainer.optimizer.zero_grad()
    with record_function('backward'):
        loss.backward()
    with record_function('optimizer'):
        clip_gradients_per_group(model, config.grad_clip_norm)
        trainer.optimizer.step()


# warm up on every batch first: compile + cuDNN autotune for each coil bucket
for batch in batches:
    step(batch)
torch.cuda.synchronize()
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for batch in batches[:6]:
    step(batch)
end.record()
torch.cuda.synchronize()
step_ms = start.elapsed_time(end) / 6

with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    for batch in batches[6:9]:
        step(batch)
    torch.cuda.synchronize()

KINDS = [
    ('fft', re.compile(r'fft|FFT|regular_fft|vector_fft|cufft', re.I)),
    ('conv/gemm (cudnn/cutlass)', re.compile(r'conv|cudnn|implicit|wgrad|dgrad|sm80_xmma|sm86_xmma|xmma|cutlass|gemm|ampere', re.I)),
    ('triton (compiled U-Net)', re.compile(r'^triton_')),
    ('norm', re.compile(r'norm|welford', re.I)),
    ('reduce', re.compile(r'reduce|Reduce', re.I)),
    ('copy/cat/index', re.compile(r'copy|Copy|cat|index|gather|scatter|roll|fill|memcpy|Memcpy|memset', re.I)),
    ('elementwise', re.compile(r'elementwise|Elementwise|unrolled|vectorized', re.I)),
]
totals = defaultdict(float)
names = defaultdict(float)
for evt in prof.key_averages():
    if evt.device_type != torch.autograd.DeviceType.CUDA:
        continue
    t = evt.self_device_time_total / 3 / 1000  # ms per step
    names[evt.key] += t
    kind = next((k for k, pattern in KINDS if pattern.search(evt.key)), 'other')
    totals[kind] += t
top = sorted(names.items(), key=lambda kv: -kv[1])[:30]
result = {'extra': extra, 'step_ms_cuda_events': step_ms, 'kernel_ms_per_step_by_kind': dict(sorted(totals.items(), key=lambda kv: -kv[1])),
          'kernel_ms_per_step_total': sum(totals.values()), 'top_kernels_ms': top,
          'peak_GiB': torch.cuda.max_memory_allocated() / 2**30}
json.dump(result, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in result.items() if k != 'top_kernels_ms'}, indent=1))
for name, ms in top[:15]:
    print(f'{ms:8.2f} ms  {name[:120]}')
