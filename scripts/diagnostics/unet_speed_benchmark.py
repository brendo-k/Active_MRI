"""Speed of the VarNet U-Net arms of reports/unet_speed_varnet2_r8, at bs 16 and 256x256, synchronized wall-clock.

Run on an idle GPU. Every time is the median over `--repeats` blocks of `--iters` steps after warm-up. A
configuration that runs out of GPU memory is recorded as null and the rest go on. cuDNN's autotuner is off, as in
training, unless `--cudnn-benchmark`; it is set once at start-up (switching it mid-process did not take effect).

  unet       one cascade's NormUnet, forward + backward: ms, GFLOP (fwd + bwd), parameters, peak memory
  varnet     the VarNet alone at 2 and 12 cascades on synthetic k-space and maps: forward + backward + Adam step
  pipeline   one real training step of the 2-cascade run's model (sensitivity net, fixed mask, VarNet, L1, backward,
             Adam) per arm, built from the base run's config; the sensitivity net's forward + backward alone (as
             trained, and with the in-place LeakyReLU it had before); the data loader alone at 8 and 12 workers
  compile    (--compile) the base and bf16 U-Net under torch.compile, and every pipeline arm with its U-Nets compiled
             as training.compile_unets does

`original` is base with every LeakyReLU in place, as before this study: fastMRI's U-Net in the cascades and the
sensitivity net's own in-place LeakyReLUs. The lite arms in speed.json and speed_cudnn_benchmark.json were benchmarked
with the code snapshot they trained on; their options are gone.

    CUDA_VISIBLE_DEVICES=0 CC=gcc CXX=g++ python scripts/diagnostics/unet_speed_benchmark.py --compile --cudnn-benchmark \
        --out reports/unet_speed_varnet2_r8/speed_compile.json
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.flop_counter import FlopCounterMode

import _path  # noqa: F401
from activemri.reconstructors.VarNet import NormUnet, VarNet
from activemri.training.builders import build_dataloaders, build_train_config
from activemri.training.trainer import build_model, compile_unets

B, COILS, H, W = 16, 16, 256, 256
PRECISIONS = {'base': 'fp32', 'bf16': 'bf16'}
PIPELINE_ARMS = {
    'original': {},
    'base': {},
    'bf16': {'recon.unet_precision': 'bf16'},
    'sens64': {'sampler.sens_resolution': 64},
    'sens64_bf16': {'sampler.sens_resolution': 64, 'recon.unet_precision': 'bf16'},
}
BASE_RUN = Path.home() / 'data/seqmri_checkpoints/unet_speed/runs/varnet2_r8_base_seed42'
# Options of the base run's config that the current code no longer has (the lite U-Net's).
REMOVED_RECON_KEYS = ('unet_top_chans', 'unet_top_convs', 'unet_skip_fusion', 'unet_fusion_levels')


def timed(step, iters, repeats, warmup=5):
    """Median ms per step, and the peak memory (GiB) over the timed steps; (None, None) if it runs out of memory."""
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        blocks = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iters):
                step()
            torch.cuda.synchronize()
            blocks.append((time.perf_counter() - start) / iters * 1e3)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None
    return statistics.median(blocks), torch.cuda.max_memory_allocated() / 2 ** 30


def fmt(ms, mem):
    return 'out of memory' if ms is None else f'{ms:7.1f} ms  peak {mem:5.2f} GiB'


def with_in_place_leaky_relu(model):
    """A copy of `model` with every LeakyReLU in place, as fastMRI's U-Net and the sensitivity net had them."""
    model = copy.deepcopy(model)
    for module in model.modules():
        if isinstance(module, nn.LeakyReLU):
            module.inplace = True
    return model


def bench_unet(args):
    x = torch.randn(B, 1, H, W, dtype=torch.complex64, device='cuda', requires_grad=True)
    models = {'fastMRI (in-place LeakyReLU)': with_in_place_leaky_relu(NormUnet(18, 4).cuda())}
    models |= {arm: NormUnet(18, 4, precision=precision).cuda() for arm, precision in PRECISIONS.items()}
    out = {}
    for name, model in models.items():
        def step():
            model(x).abs().sum().backward()

        with FlopCounterMode(display=False) as counter:
            step()
        ms, mem = timed(step, args.iters, args.repeats)
        out[name] = {'ms': ms, 'gflop': counter.get_total_flops() / 1e9,
                     'params_m': sum(p.numel() for p in model.parameters()) / 1e6, 'peak_gib': mem}
        print(f"unet  {name:30s} {ms:6.1f} ms  {out[name]['gflop']:6.1f} GFLOP  {out[name]['params_m']:.3f} M  "
              f"peak {mem:.2f} GiB", flush=True)
    return out


def bench_varnet(args):
    kspace = torch.randn(B, COILS, H, W, dtype=torch.complex64, device='cuda')
    maps = torch.randn(B, COILS, H, W, dtype=torch.complex64, device='cuda') * 0.25
    mask = (torch.rand(1, 1, H, W, device='cuda') < 0.125).float().expand(B, -1, -1, -1)
    masked = kspace * mask
    out = {}
    for cascades in (2, 12):
        for arm, precision in ({'original': 'fp32'} | PRECISIONS).items():
            model = VarNet(num_cascades=cascades, unet_precision=precision).cuda()
            if arm == 'original':
                model = with_in_place_leaky_relu(model)
            optimizer = torch.optim.Adam(model.parameters(), 1e-4)

            def step():
                loss = model(masked, maps, mask).abs().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            ms, mem = timed(step, max(2, args.iters // cascades * 2), args.repeats)
            out.setdefault(arm, {})[cascades] = {'ms': ms, 'peak_gib': mem}
            print(f'varnet {cascades:2d} cascades {arm:10s} {fmt(ms, mem)}', flush=True)
            del model, optimizer
            torch.cuda.empty_cache()
    return out


def sens_net_step(sens_net, kspace):
    def step():
        maps, _ = sens_net(kspace, mask_method='pre_network')
        maps.abs().sum().backward()
    return step


def base_run_config():
    cfg = OmegaConf.load(BASE_RUN / '.hydra' / 'config.yaml')
    for key in REMOVED_RECON_KEYS:
        cfg.recon.pop(key, None)
    return cfg


def pipeline_model(cfg, arm, compiled):
    cfg = copy.deepcopy(cfg)
    for key, value in PIPELINE_ARMS[arm].items():
        OmegaConf.update(cfg, key, value)
    model = build_model(build_train_config(cfg)).cuda().train()
    if arm == 'original':
        model = with_in_place_leaky_relu(model)
    if compiled:
        compile_unets(model)
    return model


def bench_pipeline(args):
    cfg = base_run_config()
    out = {'loader_ms_per_batch': {}}
    for workers in (8, 12):
        cfg.datasets.num_workers = workers
        loader, _ = build_dataloaders(cfg)
        batches = iter(loader)
        next(batches)
        start = time.perf_counter()
        for _ in range(args.loader_batches):
            next(batches)
        out['loader_ms_per_batch'][workers] = (time.perf_counter() - start) / args.loader_batches * 1e3
        print(f"loader {workers:2d} workers {out['loader_ms_per_batch'][workers]:7.0f} ms/batch", flush=True)
        if workers == 12:
            batch = next(batches)
        del batches, loader
    kspace, target = batch['kspace'].cuda(), batch['target'].cuda()

    for compiled in (False, True) if args.compile else (False,):
        for arm in PIPELINE_ARMS:
            if compiled and arm == 'original':
                continue
            model = pipeline_model(cfg, arm, compiled)
            optimizer = torch.optim.Adam(model.parameters(), 1e-4)

            def step():
                output = model(target, kspace)['output']
                loss = (output.abs() - (target if output.dim() == target.dim() else target.unsqueeze(1))).abs().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            name = arm + (' compiled' if compiled else '')
            ms, mem = timed(step, args.iters // 2, args.repeats)
            out.setdefault('train_step', {})[name] = {'ms': ms, 'peak_gib': mem}
            print(f'pipeline train step {name:20s} {fmt(ms, mem)}', flush=True)
            if arm == 'base' and not compiled:
                complex_kspace = torch.view_as_complex(kspace.contiguous())
                ms, _ = timed(sens_net_step(model.sensetivity_net, complex_kspace), args.iters // 2, args.repeats)
                in_place = with_in_place_leaky_relu(model.sensetivity_net)
                ms_in_place, _ = timed(sens_net_step(in_place, complex_kspace), args.iters // 2, args.repeats)
                out['sens_net_ms'] = {'as_trained': ms, 'in_place_leaky_relu': ms_in_place}
                print(f'sensitivity net fwd+bwd {ms:7.1f} ms (in-place LeakyReLU, as before: {ms_in_place:7.1f} ms)',
                      flush=True)
                del in_place
            del model, optimizer
            torch.cuda.empty_cache()
    return out


def bench_compile(args):
    x = torch.randn(B, 1, H, W, dtype=torch.complex64, device='cuda', requires_grad=True)
    out = {}
    for arm, precision in PRECISIONS.items():
        model = NormUnet(18, 4, precision=precision).cuda()
        model.unet.compile(dynamic=False)

        def step():
            model(x).abs().sum().backward()

        ms, mem = timed(step, args.iters, args.repeats, warmup=3)
        out[arm] = {'ms': ms, 'peak_gib': mem}
        print(f'compile unet {arm:10s} {fmt(ms, mem)}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--loader-batches', type=int, default=30)
    parser.add_argument('--compile', action='store_true', help='also time torch.compile (needs a C compiler: CC=gcc)')
    parser.add_argument('--cudnn-benchmark', action='store_true', help="turn on cuDNN's autotuner (training leaves it off)")
    parser.add_argument('--out', type=Path, default=Path('reports/unet_speed_varnet2_r8/speed_compile.json'))
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.manual_seed(0)

    result = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version(),
              'cudnn_benchmark': args.cudnn_benchmark,
              'batch': [B, COILS, H, W], 'unet': bench_unet(args), 'varnet': bench_varnet(args),
              'pipeline': bench_pipeline(args)}
    if args.compile:
        result['compile'] = bench_compile(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
