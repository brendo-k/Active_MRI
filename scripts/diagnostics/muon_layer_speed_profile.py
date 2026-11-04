"""Time and kernel breakdown of one training step of the Muon + LayerNorm VarNet + LOUPE (R=8, 12 cascades, bs 16, 320x320).

Wall-clock per phase (host copy, forward, backward, gradient clip, optimizer) comes from unprofiled synchronised steps, the
sensitivity net's own cost from a forward + backward through it alone. Kernels are profiled one phase at a time, so the
tables separate the forward's kernels from the backward's. Needs an idle GPU.

    CUDA_VISIBLE_DEVICES=N python scripts/diagnostics/muon_layer_speed_profile.py <out.json> [hydra overrides...]
"""
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('WANDB_MODE', 'disabled')
REPO = Path(__file__).resolve().parents[2]
out, extra = sys.argv[1], sys.argv[2:]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import torch
from hydra import compose, initialize_config_dir
from torch.profiler import ProfilerActivity, profile

from activemri.training.builders import build_dataloaders, build_train_config
from activemri.training.trainer import Sequential_Trainer, clip_gradients_per_group, gradient_stats
import train as train_module

CONTROL = ['recon=varnet', 'datasets=fastmri_local_v1_flair_knee', 'misc.resolution=[320,320]', 'sampler.accelerations=[8.0]',
           'sampler.budget_over=acquired', 'training.num_step=1', 'training.batch_size=16', 'training.num_epochs=100',
           'training.lr_step_size=90', 'training.seed=42', 'sampler.sampler_type=loupe_2d', 'sampler.loupe_vd_init_power=4',
           'sampler.anatomy_conditioning=false', 'datasets.limit_volumes=0.2', 'training.optimizer=muon', 'recon.norm=layer',
           'sampler.sens_norm=layer']
WARMUP_BATCHES, TIMED_BATCHES, PROFILED_BATCHES = 12, 8, 3
KERNEL_KINDS = [
    ('fft', re.compile(r'fft|FFT|cufft', re.I)),
    ('nchw<->nhwc transpose', re.compile(r'nchwToNhwc|nhwcToNchw', re.I)),
    ('conv/gemm', re.compile(r'conv|cudnn|implicit|wgrad|dgrad|xmma|cutlass|gemm|ampere', re.I)),
    ('triton (compiled)', re.compile(r'^triton_')),
    ('reduce', re.compile(r'reduc|sum|mean|norm', re.I)),
    ('copy/cat/index', re.compile(r'copy|cat|index|gather|scatter|roll|fill|mem', re.I)),
    ('elementwise', re.compile(r'elementwise|unrolled|vectorized|functor', re.I)),
]

work_dir = Path(os.environ.get('MLS_WORK_DIR', '/tmp/claude-2081/mls_profile'))
with initialize_config_dir(config_dir=str(REPO / 'configs'), version_base=None):
    cfg = compose(config_name='config', overrides=CONTROL + extra + [f'output.out_dir={work_dir}'])
config = build_train_config(cfg)
train_module.seed_everything(cfg.training.seed)
torch.backends.cudnn.benchmark = config.cudnn_benchmark
train_loader, val_loader = build_dataloaders(cfg)
trainer = Sequential_Trainer(config, train_loader, torch.device('cuda'), val_loader=val_loader)
model = trainer.model
model.train()
model.store_visuals = False
META = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0}

loader = iter(train_loader)
batches = [next(loader) for _ in range(WARMUP_BATCHES)]


def timed(phase, seconds, fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    seconds[phase] = time.perf_counter() - start
    return result


def step(batch, seconds=None, profiled=None):
    """One training step. `profiled` names the phase whose kernels `kernel_times` collects."""
    seconds = {} if seconds is None else seconds

    def run(phase, fn):
        if phase != profiled:
            return timed(phase, seconds, fn)
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            result = fn()
            torch.cuda.synchronize()
        for event in prof.key_averages():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                kernel_times[phase][event.key] += event.self_device_time_total / 1000
        return result

    kspace = run('to_device', lambda: trainer._load_kspace(batch))
    target = batch['target'].to(trainer.device)

    def forward():
        pred = trainer._train_forward(target, kspace, batch)
        return pred, model.loss(pred, trainer._target_dict(batch, target, kspace), META, config.loss_type)[0]

    _, loss = run('forward', forward)
    trainer.optimizer.zero_grad()
    run('backward', loss.backward)
    run('grad_stats', lambda: gradient_stats(model, config.grad_clip_norm))
    run('clip', lambda: clip_gradients_per_group(model, config.grad_clip_norm))
    run('optimizer', trainer.optimizer.step)
    return seconds


kernel_times = defaultdict(lambda: defaultdict(float))
for batch in batches:  # torch.compile, and the cuDNN autotuner for each coil bucket
    step(batch)

per_step = defaultdict(list)
for index in range(TIMED_BATCHES):
    for phase, value in step(batches[index % WARMUP_BATCHES]).items():
        per_step[phase].append(value * 1000)
phases_ms = {phase: statistics.median(values) for phase, values in per_step.items()}

sens_seconds = []
for batch in batches[:TIMED_BATCHES]:
    kspace = torch.view_as_complex(batch['kspace'].to(trainer.device).contiguous())
    torch.cuda.synchronize()
    start = time.perf_counter()
    maps, _ = model.sensetivity_net(kspace)
    maps.abs().sum().backward()
    torch.cuda.synchronize()
    sens_seconds.append(time.perf_counter() - start)
model.zero_grad()

for phase in ('to_device', 'forward', 'backward', 'optimizer'):
    for batch in batches[:PROFILED_BATCHES]:
        step(batch, profiled=phase)

report = {'extra': extra, 'phases_ms': phases_ms, 'step_ms': sum(phases_ms.values()), 'sens_net_fwd_bwd_ms': statistics.median(sens_seconds) * 1000,
          'peak_GiB': torch.cuda.max_memory_allocated() / 2**30, 'kernels': {}}
for phase, kernels in kernel_times.items():
    kinds = defaultdict(float)
    for name, total in kernels.items():
        kinds[next((kind for kind, pattern in KERNEL_KINDS if pattern.search(name)), 'other')] += total / PROFILED_BATCHES
    top = sorted(((name[:140], total / PROFILED_BATCHES) for name, total in kernels.items()), key=lambda item: -item[1])[:30]
    report['kernels'][phase] = {'by_kind_ms': dict(sorted(kinds.items(), key=lambda item: -item[1])), 'top_ms': top}
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(report, indent=1))
print(json.dumps({key: value for key, value in report.items() if key != 'kernels'}, indent=1))
for phase, table in report['kernels'].items():
    print(phase, json.dumps(table['by_kind_ms']))
    for name, ms in table['top_ms'][:12]:
        print(f'  {ms:8.1f} ms  {name[:110]}')
