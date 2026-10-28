"""Where a VarNet training step's time goes: torch.profiler over the real Sequential_Trainer.train_epoch loop.

Builds the trainer as train.py does, from scripts/launch/train_unet_speed_varnet2_r8.sh's overrides plus any given on
the command line, and runs `--skip + 2 + --active + --timed` training steps: the first `--skip` warm up (compile, cuDNN
autotune, first image logs), the next 2 warm the profiler up, `--active` are recorded, and the last `--timed` run
unprofiled for the true step time. The trainer's phases are labelled with record_function by wrapping its methods and
the module functions it calls (the trainer is not edited). The backward is split where the gradient of the sense maps
is complete: before that the loss, the final RSS and the VarNet cascades; after it the sensitivity net.

Reported over the recorded steps:
  step_ms           unprofiled (`--timed`) and profiled ms per step, and the ms per step blocked on the data loader
  gpu_busy          fraction of the recorded window with a kernel, copy or memset on the GPU
  phases            per phase (innermost label): host ms per step, GPU ms per step of the work it launched, and GPU-idle
                    ms per step while the host was in it (where the GPU starves)
  kernels           GPU ms per step by kernel category, and the 25 longest kernels
  syncs             host ms per step blocked in synchronising CUDA calls, by phase

wandb runs offline into `--out-dir`, so the logging cost stays in the profile and nothing reaches the project.
`--replay N` trains on the first N batches over and over, so the trainer's per-slice metric-mask cache is warm as it is
after the first epoch; it understates content-dependent costs (the metric mask's flood fill took 66 ms per step on the
replayed batches of head_bf16_compiled.json, 153 ms on fresh ones in bf16_compiled.json). `--val-epochs` times
validation afterwards; the first epoch includes compiling the eval graphs and filling the cache.

`--what-if` swaps in stand-ins that time a candidate speed-up (the metrics they log are meaningless):
  cached_metric_mask       the first batch's metric mask, kept in pinned memory, for every later batch: the cost of a
                           warm per-slice cache
  no_train_masked_metrics  the train loop's MaskedMetricAccumulator.update (batched masked PSNR and SSIM) does nothing
  sens_bf16                the sensitivity net's U-Net under bf16 autocast, as recon.unet_precision does the cascades'

    CUDA_VISIBLE_DEVICES=3 CC=gcc CXX=g++ python scripts/diagnostics/varnet_step_profile.py --name bf16_compiled \
        recon.unet_precision=bf16 training.compile_unets=true
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_SILENT'] = 'true'
for _pool in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_pool, '8')

import argparse
import bisect
import collections
import functools
import itertools
import json
import random
import re
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch.profiler import ProfilerActivity, profile, record_function, schedule

import _path  # noqa: F401
import activemri.training.trainer as trainer_module
from activemri.training.builders import build_dataloaders, build_train_config

REPO = Path(__file__).resolve().parents[2]
MASK = Path.home() / 'data/seqmri_checkpoints/unet_speed/static_vd_r8_mask_seed42.pt'
# scripts/launch/train_unet_speed_varnet2_r8.sh at 2 cascades, seed 42.
LAUNCHER_OVERRIDES = [
    'hydra.job.name=train_sequential', 'recon=varnet', 'recon.num_cascades=2', 'datasets=brain_knee_balanced',
    'datasets.num_workers=12', 'sampler.accelerations=[8.0]', 'sampler.sampler_type=loupe_2d',
    f'sampler.fixed_mask_path={MASK}', 'training.batch_size=16', 'training.lr=0.001', 'training.lr_scheduler=step',
    'training.lr_step_size=40', 'training.num_epochs=50', 'training.seed=42',
]
PROFILER_WARMUP = 2
WHAT_IFS = ('cached_metric_mask', 'no_train_masked_metrics', 'sens_bf16')
BACKWARD_SPLIT = 'backward/sens_net'
BACKWARD_BEFORE_SPLIT = 'backward/loss+rss+cascades'
SENSE_GRAD_MARKER = 'marker/sense_maps_grad_ready'
KERNEL_CATEGORIES = [
    ('triton (compiled U-Net)', r'^triton_'),
    ('memcpy', r'^Memcpy|^memcpy'),
    ('memset', r'^Memset|^memset'),
    ('FFT', r'fft|FFT'),
    ('conv / gemm', r'conv|Conv|cudnn|implicit_gemm|xmma|cutlass|gemm|sm80_|sm86_|wgrad|dgrad|fprop|winograd|nchwToNhwc|nhwcToNchw'),
    ('norm', r'norm|welford|Norm'),
    ('grid_sample / pool / upsample', r'grid_sampler|pool|upsample|Upsample'),
    ('reduce', r'reduce|Reduce'),
    ('elementwise', r'elementwise|vectorized|unrolled|Elementwise|index|copy|cat|CatArray'),
]
SYNC_CALLS = ('cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize', 'cudaMemcpy', 'cudaMemcpyAsync',
              'cuStreamSynchronize', 'cuMemcpyDtoH', 'cuMemcpyDtoHAsync')
LAUNCH_CALLS = re.compile(r'^(cuda|cu)(LaunchKernel|LaunchKernelEx|LaunchKernelExC|MemcpyAsync|Memcpy|MemsetAsync|'
                          r'Memset|MemcpyDtoHAsync|MemcpyHtoDAsync|LaunchCooperativeKernel|GraphLaunch)')


def labelled(name, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with record_function(name):
            return fn(*args, **kwargs)
    return wrapper


def patch_attribute(owner, attribute, name):
    setattr(owner, attribute, labelled(name, getattr(owner, attribute)))


class LabelledLoader:
    """The train loader, stopped after `steps` batches, with each step and each wait for a batch labelled; it steps the
    profiler at every step boundary and records each step's wall time."""

    def __init__(self, loader, steps, replay=0, steady_steps=0):
        self.loader = loader
        self.steps = steps
        self.replay = replay
        self.steady_steps = steady_steps
        self.overall_peak_before_steady_gib = None
        self.steady_peak_gib = None
        self.profiler = None
        self.step_seconds = []
        self.wait_seconds = []

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, attribute):
        return getattr(self.loader, attribute)

    def __iter__(self):
        batches = iter(self.loader)
        if self.replay:
            batches = itertools.cycle([next(batches) for _ in range(self.replay)])
        previous = None
        for index in range(self.steps):
            if self.steady_steps and index == self.steps - self.steady_steps:
                torch.cuda.synchronize()
                self.overall_peak_before_steady_gib = torch.cuda.max_memory_allocated() / 2 ** 30
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with record_function('loader_wait'):
                batch = next(batches)
            self.wait_seconds.append(time.perf_counter() - start)
            step = record_function('step')
            step.__enter__()
            yield batch
            step.__exit__(None, None, None)
            now = time.perf_counter()
            if previous is not None:
                self.step_seconds.append(now - previous)
            previous = now
            if self.profiler is not None:
                self.profiler.step()


def label_trainer(trainer):
    """Wrap the trainer's phases in record_function labels."""
    model = trainer.model
    for method in ('_load_kspace', '_target', '_train_forward', '_optimize', '_metric_mask', '_target_dict'):
        patch_attribute(trainer, method, method.lstrip('_'))
    patch_attribute(model, 'loss', 'loss')
    patch_attribute(model, 'step_forward', 'forward/step_forward (DC inputs, zero-filled, vis copies)')
    patch_attribute(model.reconstructor, 'forward', 'forward/varnet')
    if model.mask_policy is not None:
        patch_attribute(model.mask_policy, 'forward', 'forward/mask_policy')
    for index, cascade in enumerate(model.reconstructor.cascades):
        patch_attribute(cascade.model, 'forward', f'forward/varnet/cascade{index}_unet')

    sens_forward = model.sensetivity_net.forward

    def sens_net(*args, **kwargs):
        with record_function('forward/sens_net'):
            maps, sense_mask = sens_forward(*args, **kwargs)
        if maps.requires_grad:
            maps.register_hook(_mark_sense_grad)
        return maps, sense_mask
    model.sensetivity_net.forward = sens_net

    patch_attribute(trainer.optimizer, 'step', 'optimizer.step')
    patch_attribute(trainer.optimizer, 'zero_grad', 'optimizer.zero_grad')
    for function in ('gradient_stats', 'clip_gradients_per_group', 'nonfinite_gradient_params', 'zero_coil_fraction',
                     '_apply_signal_mask', 'metric_mask_from_real_kspace'):
        patch_attribute(trainer_module, function, function.lstrip('_'))
    patch_attribute(trainer_module.MaskedMetricAccumulator, 'update', 'masked_metrics.update')
    patch_attribute(trainer_module._AnatomyStats, 'update', 'anatomy_stats.update')
    patch_attribute(trainer_module.wandb, 'log', 'wandb.log')
    patch_attribute(torch.Tensor, 'backward', 'backward')

    build_metrics = trainer_module.build_metrics

    def labelled_metrics(*args, **kwargs):
        metrics = build_metrics(*args, **kwargs)
        patch_attribute(metrics, 'update', 'metrics.update (PSNR, SSIM)')
        return metrics
    trainer_module.build_metrics = labelled_metrics


class BF16Autocast(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.inner(x).float()


def apply_what_ifs(trainer, what_ifs):
    if 'cached_metric_mask' in what_ifs:
        metric_mask = trainer._metric_mask
        cache = {}

        def cached(batch):
            if 'mask' not in cache:
                cache['mask'] = metric_mask(batch).cpu().pin_memory()
            return cache['mask'].to(trainer.device, non_blocking=True)
        trainer._metric_mask = cached
    if 'no_train_masked_metrics' in what_ifs:
        trainer_module.MaskedMetricAccumulator.update = lambda self, *args, **kwargs: None
    if 'sens_bf16' in what_ifs:
        sens_net = trainer.model.sensetivity_net
        sens_net.model = BF16Autocast(sens_net.model)


def _mark_sense_grad(grad):
    with record_function(SENSE_GRAD_MARKER):
        pass
    return grad


def load_trace(path):
    events = json.loads(Path(path).read_text())['traceEvents']
    return [e for e in events if e.get('ph') == 'X']


def main_thread_intervals(events):
    """Labelled intervals (start, end, name) on the host thread that runs the step, with the backward split at the
    sense maps' gradient."""
    annotations = [e for e in events if e.get('cat') == 'user_annotation']
    step_threads = collections.Counter(e['tid'] for e in annotations if e['name'] == 'step')
    main_tid = step_threads.most_common(1)[0][0]
    intervals = [(e['ts'], e['ts'] + e['dur'], e['name']) for e in annotations
                 if e['tid'] == main_tid and not e['name'].startswith('ProfilerStep')]
    markers = sorted(e['ts'] for e in annotations if e['name'] == SENSE_GRAD_MARKER)
    for start, end, name in [i for i in intervals if i[2] == 'backward']:
        split = next((m for m in markers if start <= m <= end), None)
        if split is not None:
            intervals += [(start, split, BACKWARD_BEFORE_SPLIT), (split, end, BACKWARD_SPLIT)]
    return intervals, main_tid


def innermost_timeline(intervals):
    """Disjoint segments (start, end, name) labelled by the innermost interval covering them."""
    points = sorted({t for start, end, _ in intervals for t in (start, end)})
    segments = []
    by_start = sorted(intervals)
    for left, right in zip(points, points[1:]):
        middle = (left + right) / 2
        covering = [(end - start, name) for start, end, name in by_start if start <= middle < end]
        if covering:
            segments.append((left, right, min(covering)[1]))
    return segments


def label_at(segments, starts, t):
    index = bisect.bisect_right(starts, t) - 1
    if index >= 0 and segments[index][0] <= t < segments[index][1]:
        return segments[index][2]
    return 'unlabelled'


def kernel_category(name):
    return next((category for category, pattern in KERNEL_CATEGORIES if re.search(pattern, name)), 'other')


def union_length(spans):
    total, current_start, current_end = 0.0, None, None
    for start, end in sorted(spans):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + (current_end - current_start if current_end is not None else 0.0)


def analyse(trace_path, active_steps):
    events = load_trace(trace_path)
    intervals, main_tid = main_thread_intervals(events)
    steps = sorted((s, e) for s, e, name in intervals if name == 'step')
    waits = sorted((s, e) for s, e, name in intervals if name == 'loader_wait')
    window = (min(s for s, _ in steps + waits), max(e for _, e in steps))
    segments = innermost_timeline(intervals)
    segment_starts = [s for s, _, _ in segments]

    gpu = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')
           and window[0] <= e['ts'] < window[1]]
    launches = {e['args']['correlation']: e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver')
                and 'correlation' in e.get('args', {}) and LAUNCH_CALLS.match(e['name'])}

    host_ms = collections.Counter()
    for start, end, name in segments:
        lo, hi = max(start, window[0]), min(end, window[1])
        if hi > lo:
            host_ms[name] += (hi - lo) / 1e3

    gpu_ms_by_phase, gpu_ms_by_category, gpu_ms_by_kernel = collections.Counter(), collections.Counter(), collections.Counter()
    for kernel in gpu:
        launch = launches.get(kernel.get('args', {}).get('correlation'))
        phase = label_at(segments, segment_starts, launch['ts']) if launch else 'unlabelled'
        gpu_ms_by_phase[phase] += kernel['dur'] / 1e3
        gpu_ms_by_category[kernel_category(kernel['name'])] += kernel['dur'] / 1e3
        gpu_ms_by_kernel[kernel['name'][:110]] += kernel['dur'] / 1e3

    busy = sorted((k['ts'], k['ts'] + k['dur']) for k in gpu)
    merged = []
    for start, end in busy:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps = [(window[0], merged[0][0])] + [(a[1], b[0]) for a, b in zip(merged, merged[1:])] + [(merged[-1][1], window[1])]
    idle_ms = collections.Counter()
    for gap_start, gap_end in gaps:
        if gap_end <= gap_start:
            continue
        index = max(0, bisect.bisect_right(segment_starts, gap_start) - 1)
        covered = 0.0
        while index < len(segments) and segments[index][0] < gap_end:
            lo, hi = max(segments[index][0], gap_start), min(segments[index][1], gap_end)
            if hi > lo:
                idle_ms[segments[index][2]] += (hi - lo) / 1e3
                covered += hi - lo
            index += 1
        idle_ms['unlabelled'] += max(0.0, gap_end - gap_start - covered) / 1e3

    sync_ms = collections.Counter()
    for e in events:
        if e.get('cat') in ('cuda_runtime', 'cuda_driver') and e['tid'] == main_tid and e['name'] in SYNC_CALLS \
                and window[0] <= e['ts'] < window[1] and e['dur'] > 50:
            sync_ms[(label_at(segments, segment_starts, e['ts']), e['name'])] += e['dur'] / 1e3

    window_ms = (window[1] - window[0]) / 1e3
    per_step = lambda counter: {k: v / active_steps for k, v in counter.most_common()}  # noqa: E731
    phases = sorted(set(host_ms) | set(gpu_ms_by_phase) | set(idle_ms),
                    key=lambda p: -(host_ms.get(p, 0) + gpu_ms_by_phase.get(p, 0)))
    return {
        'window_ms_per_step': window_ms / active_steps,
        'gpu_busy_fraction': union_length([(k['ts'], k['ts'] + k['dur']) for k in gpu]) / (window[1] - window[0]),
        'gpu_ms_per_step': sum(k['dur'] for k in gpu) / 1e3 / active_steps,
        'phases': {p: {'host_ms': host_ms.get(p, 0) / active_steps, 'gpu_ms': gpu_ms_by_phase.get(p, 0) / active_steps,
                       'gpu_idle_ms': idle_ms.get(p, 0) / active_steps} for p in phases},
        'kernel_categories_ms': per_step(gpu_ms_by_category),
        'top_kernels_ms': dict(list(per_step(gpu_ms_by_kernel).items())[:25]),
        'syncs_ms': {f'{phase} | {call}': ms for (phase, call), ms in per_step(sync_ms).items()},
    }


def print_report(name, result):
    print(f"\n=== {name} ===")
    print(f"step ms: unprofiled {result['step_ms_unprofiled']:.1f}, profiled {result['step_ms_profiled']:.1f}, "
          f"loader wait {result['loader_wait_ms']:.1f} (median over profiled + timed steps)")
    print(f"GPU busy {result['gpu_busy_fraction'] * 100:.1f}% of the recorded window; "
          f"GPU work {result['gpu_ms_per_step']:.1f} ms/step")
    print(f"\n{'phase (innermost label)':58s} {'host ms':>8s} {'GPU ms':>8s} {'GPU idle':>9s}")
    for phase, row in result['phases'].items():
        print(f"{phase:58s} {row['host_ms']:8.1f} {row['gpu_ms']:8.1f} {row['gpu_idle_ms']:9.1f}")
    print('\nGPU ms/step by kernel category')
    for category, ms in result['kernel_categories_ms'].items():
        print(f'  {category:40s} {ms:7.1f}')
    print('\ntop kernels, GPU ms/step')
    for kernel, ms in result['top_kernels_ms'].items():
        print(f'  {ms:7.2f}  {kernel}')
    print('\nhost blocked in synchronising CUDA calls (> 50 us), ms/step')
    for key, ms in result['syncs_ms'].items():
        print(f'  {ms:7.2f}  {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True, help='name of this configuration in the output')
    parser.add_argument('--skip', type=int, default=12, help='warm-up steps before the profiler starts')
    parser.add_argument('--active', type=int, default=10, help='profiled steps')
    parser.add_argument('--timed', type=int, default=15, help='unprofiled steps after the profile, for the step time')
    parser.add_argument('--out-dir', type=Path, required=True, help='where the trace, wandb files and JSON go')
    parser.add_argument('--what-if', nargs='*', default=[], choices=WHAT_IFS, help='timing stand-ins (see above)')
    parser.add_argument('--replay', type=int, default=0,
                        help='train on the first N batches over and over, so per-slice caches are warm after N steps')
    parser.add_argument('--val-epochs', type=int, default=0, help='then time this many validation epochs')
    args, overrides = parser.parse_known_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.out_dir / args.name
    with initialize_config_dir(config_dir=str(REPO / 'configs'), version_base=None):
        cfg = compose('config', overrides=LAUNCHER_OVERRIDES + [f'output.out_dir={run_dir}'] + overrides)
    config = build_train_config(cfg)
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    train_loader, val_loader = build_dataloaders(cfg)

    steps = args.skip + PROFILER_WARMUP + args.active + args.timed
    loader = LabelledLoader(train_loader, steps, args.replay, steady_steps=args.timed)
    trainer = trainer_module.Sequential_Trainer(config, loader, torch.device(config.device), val_loader=val_loader)
    apply_what_ifs(trainer, args.what_if)
    label_trainer(trainer)

    trace_path = args.out_dir / f'{args.name}_trace.json'
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=args.skip, warmup=PROFILER_WARMUP, active=args.active, repeat=1),
                 on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path))) as profiler:
        loader.profiler = profiler
        trainer.train_epoch()

    first_profiled = args.skip + PROFILER_WARMUP - 1
    profiled = loader.step_seconds[first_profiled:first_profiled + args.active]
    timed = loader.step_seconds[first_profiled + args.active + 1:]
    steady_peak_gib = torch.cuda.max_memory_allocated() / 2 ** 30
    overall_peak_gib = max(steady_peak_gib, loader.overall_peak_before_steady_gib or 0.0)
    result = {'name': args.name, 'overrides': overrides, 'what_ifs': args.what_if, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'step_ms_unprofiled': statistics.median(timed) * 1e3, 'step_ms_profiled': statistics.median(profiled) * 1e3,
              'loader_wait_ms': statistics.median(loader.wait_seconds[args.skip:]) * 1e3,
              'step_ms_all': [s * 1e3 for s in loader.step_seconds],
              'peak_gib': overall_peak_gib,
              'overall_peak_before_steady_gib': loader.overall_peak_before_steady_gib,
              'steady_peak_gib': steady_peak_gib}
    result['val_epoch_s'] = [trainer.val_epoch()[2] for _ in range(args.val_epochs)]
    result |= analyse(trace_path, args.active)
    (args.out_dir / f'{args.name}.json').write_text(json.dumps(result, indent=2))
    print_report(args.name, result)
    if result['val_epoch_s']:
        print('\nvalidation epoch s: ' + ', '.join(f'{s:.1f}' for s in result['val_epoch_s']))


if __name__ == '__main__':
    main()
