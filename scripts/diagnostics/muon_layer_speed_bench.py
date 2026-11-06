"""Seconds per training step and per validation pass of the Muon + LayerNorm VarNet + LOUPE, per arm, on one idle GPU.

Each arm runs in its own process, from its own code snapshot, through the trainer's `train_epoch` and `val_epoch`
(metrics and logging as in a run): a warm-up of WARMUP_BATCHES batches (torch.compile, the cuDNN autotuner for each coil
bucket), then TIMED_BATCHES batches and one validation pass, on 20% of the data.

    python scripts/diagnostics/muon_layer_speed_bench.py --out bench.json --before <repo> --after <repo>
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WARMUP_BATCHES = 14
TIMED_BATCHES = 24
CONTROL = ['recon=varnet', 'datasets=fastmri_local_v1_flair_knee', 'misc.resolution=[320,320]', 'sampler.accelerations=[8.0]',
           'sampler.budget_over=acquired', 'training.num_step=1', 'training.batch_size=16', 'training.num_epochs=100',
           'training.lr_step_size=90', 'training.seed=42', 'sampler.sampler_type=loupe_2d', 'sampler.loupe_vd_init_power=4',
           'sampler.anatomy_conditioning=false', 'datasets.limit_volumes=0.2', 'training.optimizer=muon', 'recon.norm=layer',
           'sampler.sens_norm=layer']
FAST = ['training.fused_adam=true', 'training.channels_last_unets=sens']
ARMS = {  # name: (which repo, hydra overrides)
    'control': ('before', []),
    'exact': ('after', FAST),
    'exact_16_channels': ('after', ['training.fused_adam=true', 'training.channels_last_unets=true', 'recon.chans=16']),
}


def run_arm(name, repo, out_dir):
    os.environ.setdefault('WANDB_MODE', 'disabled')
    sys.path.insert(0, repo)
    os.chdir(repo)
    import torch
    from hydra import compose, initialize_config_dir
    from activemri.training.builders import build_dataloaders, build_train_config
    from activemri.training.trainer import Sequential_Trainer
    import train as train_module

    overrides = CONTROL + ARMS[name][1] + [f'output.out_dir={out_dir}/{name}']
    with initialize_config_dir(config_dir=f'{repo}/configs', version_base=None):
        cfg = compose(config_name='config', overrides=overrides)
    config = build_train_config(cfg)
    train_module.seed_everything(cfg.training.seed)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    train_loader, val_loader = build_dataloaders(cfg)
    trainer = Sequential_Trainer(config, train_loader, torch.device('cuda'), val_loader=val_loader)
    batches = len(train_loader)

    trainer.options.limit_batches = (WARMUP_BATCHES + 0.5) / batches
    trainer.epoch = 0
    trainer.train_epoch()
    trainer.val_epoch()
    torch.cuda.synchronize()
    if ARMS[name][0] == 'after':  # _train_loop does this after its first epoch
        gc.collect()
        gc.freeze()
    torch.cuda.reset_peak_memory_stats()
    trainer.options.limit_batches = (TIMED_BATCHES + 0.5) / batches
    trainer.epoch = 1  # not the last epoch, so visualize_every applies as in a run
    start = time.perf_counter()
    trainer.train_epoch()
    torch.cuda.synchronize()
    train_s = time.perf_counter() - start
    start = time.perf_counter()
    trainer.val_epoch()
    torch.cuda.synchronize()
    return {'arm': name, 'repo': repo, 'overrides': ARMS[name][1], 'step_s': train_s / TIMED_BATCHES,
            'val_pass_s': time.perf_counter() - start, 'val_batches': len(val_loader), 'steady_peak_GiB': torch.cuda.max_memory_allocated() / 2**30,
            'gpu': torch.cuda.get_device_name()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=sorted(ARMS))
    parser.add_argument('--repo')
    parser.add_argument('--before')
    parser.add_argument('--after')
    parser.add_argument('--out')
    parser.add_argument('--work-dir', default='/tmp/claude-2081/muon_layer_speed_bench')
    args = parser.parse_args()
    if args.arm:
        print('RESULT ' + json.dumps(run_arm(args.arm, args.repo, args.work_dir)), flush=True)
        return
    results = []
    for name, (side, _) in ARMS.items():
        repo = args.before if side == 'before' else args.after
        proc = subprocess.run([sys.executable, __file__, '--arm', name, '--repo', repo, '--work-dir', args.work_dir],
                              capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': repo})
        lines = [line for line in proc.stdout.splitlines() if line.startswith('RESULT ')]
        results.append(json.loads(lines[-1][len('RESULT '):]) if lines else {'arm': name, 'error': proc.stderr[-2000:]})
        print(json.dumps(results[-1])[:300], flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
