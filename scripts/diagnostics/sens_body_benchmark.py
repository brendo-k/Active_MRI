"""Forward/backward benchmark using a captured real sensitivity U-Net batch.

Uses identical weights, inputs and upstream gradients for every arm; no optimizer
steps. This checks local numerical agreement, not final reconstruction quality.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '8')
import _path  # noqa: F401
import torch
from activemri.reconstructors.unet_backbone import Unet


def relative_error(a, b):
    a, b = a.float(), b.float()
    return {'relative_l2': float((a-b).norm() / b.norm().clamp_min(1e-20)),
            'max_abs': float((a-b).abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', default=['default', 'nchw', 'channels_last', 'chunk32', 'chunk64', 'chunk128'])
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--cudnn-limit', type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = args.cudnn_limit
    payload = torch.load(args.capture, weights_only=True)
    source = payload['input'].cuda()
    gradient = payload['grad'].cuda()
    results = {'shape': list(source.shape), 'torch': torch.__version__,
               'gpu': torch.cuda.get_device_name(), 'arms': {}}
    reference_y, reference_g = None, None
    for arm in args.arms:
        model = Unet(2, 2, chans=8).cuda().train()
        model.load_state_dict(payload['state'])
        x = source.clone()
        chunk = int(arm.removeprefix('chunk')) if arm.startswith('chunk') else 0
        if arm in ('group', 'moments'):
            from sens_layout_candidates import replace_sensitivity_norms
            replace_sensitivity_norms(model, arm)
        if arm in ('channels_last', 'group', 'moments'):
            model.to(memory_format=torch.channels_last)
            x = x.contiguous(memory_format=torch.channels_last)
        torch._dynamo.maybe_mark_dynamic(x, 0)
        options = {'layout_optimization': False} if arm == 'nchw' else {}
        model.compile(dynamic=False, options=options)
        def step():
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = torch.cat([model(part) for part in x.split(chunk)]) if chunk else model(x)
            y.backward(gradient)
            return y
        start = time.perf_counter()
        for _ in range(5):
            y = step()
        torch.cuda.synchronize()
        warmup = time.perf_counter() - start
        grads = torch.cat([p.grad.detach().flatten() for p in model.parameters()])
        if reference_y is None:
            reference_y, reference_g = y.detach().clone(), grads.clone()
        errors = {'output': relative_error(y.detach(), reference_y),
                  'parameter_grad': relative_error(grads, reference_g)}
        del y, grads
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(args.steps):
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
            step()
            end_event.record()
            end_event.synchronize()
            timings.append(start_event.elapsed_time(end_event))
        row = {'median_ms': statistics.median(timings), 'all_ms': timings, 'warmup_s': warmup,
               'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30, 'errors_vs_first_arm': errors}
        results['arms'][arm] = row
        args.out.write_text(json.dumps(results, indent=2))
        print(arm, json.dumps(row), flush=True)
        del model, x
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
