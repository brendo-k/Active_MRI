"""Diagnostic-only extensions to varnet_step_profile; production code is untouched.

Example: CUDA_VISIBLE_DEVICES=2 python scripts/diagnostics/varnet_throughput_probe.py \
  --sens-layout nchw --name nchw --replay 4 --out-dir reports/varnet_throughput_20261026
Remaining arguments go to varnet_step_profile.py. Capture saves one real normalized
coil batch, upstream gradient, and initial U-Net weights for isolated experiments.
"""
import argparse
import sys
from pathlib import Path

import varnet_step_profile as base
import torch


def install_fused_shifts():
    """Fuse the two spatial permutations; no arithmetic or precision change."""
    @torch.compile(dynamic=False)
    def roll_real(x, shifts, dims):
        return torch.roll(x, shifts, dims)

    def wrap(original, inverse):
        def shift(x, dim=None):
            if not x.is_complex():
                return original(x, dim=dim)
            dims = tuple(range(x.ndim)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
            dims = tuple(d % x.ndim for d in dims)
            shifts = tuple((-1 if inverse else 1) * (x.shape[d] // 2) for d in dims)
            return torch.view_as_complex(roll_real(torch.view_as_real(x), shifts, dims))
        return shift
    torch.fft.fftshift = wrap(torch.fft.fftshift, False)
    torch.fft.ifftshift = wrap(torch.fft.ifftshift, True)


class BodyProbe(torch.nn.Module):
    def __init__(self, inner, chunk, capture):
        super().__init__()
        self.inner = inner
        self.chunk = chunk
        self.capture = capture

    def forward(self, x):
        capture = self.capture
        self.capture = None
        if capture:
            payload = {"input": x.detach().cpu(),
                       "state": {k: v.detach().cpu().clone() for k, v in self.inner.state_dict().items()}}
        if self.chunk:
            y = torch.cat([self.inner(part) for part in x.split(self.chunk)], dim=0)
        else:
            y = self.inner(x)
        if capture:
            def save(grad):
                payload["grad"] = grad.detach().cpu()
                torch.save(payload, capture)
                return grad
            y.register_hook(save)
        return y


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--sens-layout', choices=['default', 'nchw'], default='default')
    parser.add_argument('--sens-chunk', type=int, default=0)
    parser.add_argument('--capture-sens', type=Path)
    parser.add_argument('--fused-shifts', action='store_true')
    parser.add_argument('--sens-norm', choices=['instance', 'group', 'moments'], default='instance')
    parser.add_argument('--cudnn-limit', type=int, default=10)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    torch.backends.cudnn.benchmark_limit = args.cudnn_limit
    if args.fused_shifts:
        install_fused_shifts()
    original = base.apply_what_ifs

    def configure(trainer, what_ifs):
        original(trainer, what_ifs)
        sens = trainer.model.sensetivity_net
        if args.sens_norm != 'instance':
            from sens_layout_candidates import replace_sensitivity_norms, ChannelsLastBody
            replace_sensitivity_norms(sens.model, args.sens_norm)
            sens.model.to(memory_format=torch.channels_last)
            sens.model.compile(dynamic=False)
            sens.model = ChannelsLastBody(sens.model)
        if args.sens_layout == 'nchw':
            # Replaces Module.compile's lazy callable before its first execution.
            sens.model.compile(dynamic=False, options={'layout_optimization': False})
        if args.sens_chunk or args.capture_sens:
            sens.model = BodyProbe(sens.model, args.sens_chunk, args.capture_sens)
    base.apply_what_ifs = configure

    original_profile = base.profile
    def detailed_profile(*a, **kw):
        callback = kw['on_trace_ready']
        def ready(p):
            callback(p)
            output = Path(remaining[remaining.index('--out-dir') + 1])
            name = remaining[remaining.index('--name') + 1]
            rows = []
            for event in p.key_averages(group_by_input_shape=True):
                rows.append({'op': event.key, 'shapes': event.input_shapes,
                             'calls': event.count, 'self_cuda_us': event.self_device_time_total,
                             'cuda_us': event.device_time_total, 'cpu_us': event.cpu_time_total})
            import json
            (output / f'{name}_operators.json').write_text(json.dumps(rows, indent=2))
            (output / f'{name}_probe.json').write_text(json.dumps(vars(args), default=str, indent=2))
        kw.update(record_shapes=True, on_trace_ready=ready)
        return original_profile(*a, **kw)
    base.profile = detailed_profile
    base.main()


if __name__ == '__main__':
    main()
