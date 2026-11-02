"""Attribute traced GPU kernels to training phases, separating layout copies."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import varnet_step_profile as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    output = {}
    for path in args.directory.glob('*_trace.json'):
        events = base.load_trace(path)
        intervals, _ = base.main_thread_intervals(events)
        segments = base.innermost_timeline(intervals)
        starts = [s for s, _, _ in segments]
        steps = sum(name == 'step' for _, _, name in intervals)
        launches = {e['args']['correlation']: e for e in events
                    if e.get('cat') in ('cuda_runtime', 'cuda_driver')
                    and 'correlation' in e.get('args', {}) and base.LAUNCH_CALLS.match(e['name'])}
        phases = collections.defaultdict(collections.Counter)
        kernels = collections.Counter()
        for event in events:
            if event.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
                continue
            launch = launches.get(event.get('args', {}).get('correlation'))
            phase = base.label_at(segments, starts, launch['ts']) if launch else 'unlabelled'
            name = event['name']
            category = base.kernel_category(name)
            if 'nchw' in name.lower() and 'nhwc' in name.lower() and 'kernel' in name.lower():
                category = 'layout conversion'
            elif 'roll_cuda_kernel' in name or 'fused_roll_' in name:
                category = 'FFT shift / roll'
            phases[phase][category] += event['dur'] / 1000 / steps
            kernels[category] += event['dur'] / 1000 / steps
        output[path.stem] = {'steps': steps, 'kernel_ms_per_step': dict(kernels),
                             'phase_kernel_ms_per_step': {k: dict(v) for k, v in phases.items()}}
    (args.directory / 'kernel_attribution.json').write_text(json.dumps(output, indent=2))
    paths = ['activemri/reconstructors/SensetivityModel_mc.py', 'activemri/reconstructors/unet_backbone.py',
             'activemri/reconstructors/VarNet.py', 'activemri/training/trainer.py', 'configs/config.yaml',
             'configs/recon/varnet.yaml', 'scripts/diagnostics/varnet_step_profile.py',
             'scripts/diagnostics/varnet_throughput_probe.py']
    provenance = {'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
                  'baseline_overrides': base.LAUNCHER_OVERRIDES}
    (args.directory / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps({k: v['kernel_ms_per_step'] for k, v in output.items()}, indent=2))


if __name__ == '__main__':
    main()
