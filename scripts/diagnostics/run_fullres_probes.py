"""Run full-loop full-resolution candidates serially and retain GPU-load evidence.

This runner never pauses or modifies other jobs. All runs use GPU selected by
CUDA_VISIBLE_DEVICES. Each subprocess is bounded and its complete output saved.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def gpu_state():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                             '--format=csv'], capture_output=True, text=True)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', default=['baseline_before', 'autotune', 'group', 'moments', 'fused', 'baseline_after'])
    parser.add_argument('--prefix', default='fullres_', help='Prefix for run names and suite metadata (default: fullres_).')
    parser.add_argument('--suite-name', default='fullres_suite.json', help='Suite metadata filename.')
    parser.add_argument('--skip', type=int, default=10)
    parser.add_argument('--active', type=int, default=6)
    parser.add_argument('--timed', type=int, default=20)
    parser.add_argument('--replay', type=int, default=1)
    parser.add_argument('--extra-overrides', nargs='*', default=[])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    extra = {'autotune': ['--cudnn-limit', '0'], 'group': ['--sens-norm', 'group'],
             'moments': ['--sens-norm', 'moments'], 'fused': ['--fused-shifts'],
             'combined': ['--sens-norm', 'moments', '--fused-shifts']}
    summary = []
    for arm in args.arms:
        name = args.prefix + arm
        command = [sys.executable, 'scripts/diagnostics/varnet_throughput_probe.py',
                   '--name', name, '--skip', str(args.skip), '--active', str(args.active),
                   '--timed', str(args.timed), '--replay', str(args.replay),
                   '--out-dir', str(args.out_dir), 'datasets.num_workers=4'] + extra.get(arm, []) + args.extra_overrides
        row = {'arm': arm, 'command': command, 'gpu_before': gpu_state(), 'started': time.time()}
        print('Starting', arm, flush=True)
        with (args.out_dir/f'{name}.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=420)
            except BaseException:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                raise
        row.update(returncode=process.returncode, elapsed_s=time.time()-row['started'], gpu_after=gpu_state())
        summary.append(row)
        (args.out_dir/args.suite_name).write_text(json.dumps(summary, indent=2))
        print('Finished', arm, row['returncode'], round(row['elapsed_s'], 1), flush=True)
        if process.returncode:
            print((args.out_dir/f'{name}.log').read_text()[-4000:], flush=True)


if __name__ == '__main__':
    main()
