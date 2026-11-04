"""Seconds per training step and per validation pass of the R=8 VarNet+LOUPE trainer, per arm, on one idle GPU.

Each variant runs in its own process: it builds Sequential_Trainer exactly as train.py does (20% of the local v1
FLAIR+knee data, 12 cascades, bs 16), runs one warm-up epoch of WARMUP_BATCHES batches (torch.compile and the cuDNN
autotuner for each coil bucket), then times TIMED_BATCHES training batches through `train_epoch` (the trainer's own
loop, metrics and all) and one validation pass.

    python scripts/diagnostics/varnet_train_speed_bench.py --out reports/varnet_train_speed/bench.json
    python scripts/diagnostics/varnet_train_speed_bench.py --variant fused  # one variant, prints its JSON line
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WARMUP_BATCHES = 14
TIMED_BATCHES = 24
BASE = ['recon=varnet', 'datasets=fastmri_local_v1_flair_knee', 'misc.resolution=[320,320]', 'sampler.accelerations=[8.0]',
        'sampler.budget_over=acquired', 'training.num_step=1', 'training.batch_size=16', 'training.num_epochs=40',
        'training.lr_step_size=36', 'training.seed=42', 'sampler.sampler_type=loupe_2d', 'sampler.loupe_vd_init_power=4',
        'sampler.anatomy_conditioning=false', 'datasets.limit_volumes=0.2',
        'sampler.sense_mask_method=pre_network',  # this and the rest: the defaults when the study ran
        'training.optimizer=adam', 'recon.norm=instance', 'sampler.sens_norm=instance', 'training.fused_adam=false',
        'training.channels_last_unets=false']
EVERY_BATCH = ['training.train_metrics_every=1', 'training.visualize_every=1']  # the defaults before the study
SWITCHES = ['training.train_metrics_every=10', 'training.visualize_every=10']
EAGER = ['recon.fused_coil_ops=false']
FUSED = ['recon.fused_coil_ops=true']
NHWC = {'TORCHINDUCTOR_FORCE_LAYOUT_OPT': '1'}  # inductor's channels-last layout for the compiled U-Nets (its heuristic skips them)
VARIANTS = {  # name: (hydra overrides, environment)
    'control': (EVERY_BATCH + EAGER, {}),
    'switches': (SWITCHES + EAGER, {}),
    'fused': (SWITCHES + FUSED, {}),
    'fused_nhwc': (SWITCHES + FUSED, NHWC),
    'fused_norm_none': (SWITCHES + FUSED + ['recon.norm=none', 'sampler.sens_norm=none'], {}),
    'fused_norm_none_nhwc': (SWITCHES + FUSED + ['recon.norm=none', 'sampler.sens_norm=none'], NHWC),
    'fused_norm_layer': (SWITCHES + FUSED + ['recon.norm=layer', 'sampler.sens_norm=layer'], {}),
    'fused_norm_chln': (SWITCHES + FUSED + ['recon.norm=channel_layer', 'sampler.sens_norm=channel_layer'], {}),
    'fused_norm_chln_nhwc': (SWITCHES + FUSED + ['recon.norm=channel_layer', 'sampler.sens_norm=channel_layer'], NHWC),
    'fused_muon': (SWITCHES + FUSED + ['training.optimizer=muon'], {}),
}


def run_variant(name, out_dir):
    os.environ.setdefault('WANDB_MODE', 'disabled')
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    import torch
    from hydra import compose, initialize_config_dir
    from activemri.training.builders import build_dataloaders, build_train_config
    from activemri.training.trainer import Sequential_Trainer
    import train as train_module

    overrides = BASE + VARIANTS[name][0] + [f'output.out_dir={out_dir}/{name}']
    with initialize_config_dir(config_dir=str(REPO / 'configs'), version_base=None):
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
    trainer.options.limit_batches = (TIMED_BATCHES + 0.5) / batches
    trainer.epoch = 1  # not the last epoch, so visualize_every applies as in a real run
    start = time.perf_counter()
    trainer.train_epoch()
    torch.cuda.synchronize()
    train_s = time.perf_counter() - start
    start = time.perf_counter()
    trainer.val_epoch()
    torch.cuda.synchronize()
    val_s = time.perf_counter() - start
    return {'variant': name, 'overrides': VARIANTS[name][0], 'env': VARIANTS[name][1], 'step_s': train_s / TIMED_BATCHES, 'val_pass_s': val_s,
            'val_batches': len(val_loader), 'train_batches_per_epoch': batches,
            'peak_GiB': torch.cuda.max_memory_allocated() / 2**30, 'gpu': torch.cuda.get_device_name()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=sorted(VARIANTS))
    parser.add_argument('--variants', nargs='*', default=list(VARIANTS))
    parser.add_argument('--out', default='reports/varnet_train_speed/bench.json')
    parser.add_argument('--work-dir', default='/tmp/claude-2081/varnet_train_speed_bench')
    args = parser.parse_args()
    if args.variant:
        print('RESULT ' + json.dumps(run_variant(args.variant, args.work_dir)), flush=True)
        return
    results = []
    for name in args.variants:
        proc = subprocess.run([sys.executable, __file__, '--variant', name, '--work-dir', args.work_dir],
                              capture_output=True, text=True, env={**os.environ, **VARIANTS[name][1]})
        lines = [l for l in proc.stdout.splitlines() if l.startswith('RESULT ')]
        if proc.returncode or not lines:
            results.append({'variant': name, 'error': proc.stderr[-2000:]})
        else:
            results.append(json.loads(lines[-1][len('RESULT '):]))
        print(json.dumps(results[-1])[:400], flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
