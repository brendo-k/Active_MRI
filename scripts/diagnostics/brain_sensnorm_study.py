"""Matched full brain-only VarNet training, durable supervision, and final evaluation.

Launch creates a frozen code/config snapshot. Checkpoints explicitly identify the
normalization arm; load_study_model restores that arm before loading its weights.
Only sensitivity normalization/layout differs. No fused FFT substitutions.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_SILENT'] = 'true'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '8')

import _path  # noqa: F401
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from activemri.training.builders import build_dataloaders, build_train_config
from activemri.training.trainer import Sequential_Trainer, build_model, compile_unets, _apply_signal_mask
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.metrics import MaskedMetricAccumulator
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch
from sens_layout_candidates import replace_sensitivity_norms, ChannelsLastBody

REPO = Path(__file__).resolve().parents[2]
MASK = Path.home()/'data/seqmri_checkpoints/unet_speed/static_vd_r8_mask_seed42.pt'
ARMS = ('baseline', 'moments')


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str)+'\n')
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure(root, seed, arm):
    with initialize_config_dir(config_dir=str(REPO/'configs'), version_base=None):
        return compose('config', overrides=[
            'recon=varnet', 'recon.num_cascades=12', 'recon.chans=18', 'recon.pools=4',
            'recon.unet_precision=bf16', 'recon.checkpoint_cascades=false',
            'datasets=brain', 'datasets.train_anatomies=[brain]',
            'datasets.num_workers=4', 'sampler.sampler_type=loupe_2d',
            'sampler.accelerations=[8.0]', 'sampler.budget_over=grid',
            f'sampler.fixed_mask_path={root / "mask.pt"}',
            'sampler.sens_precision=bf16', 'sampler.sens_resolution=null',
            'training.compile_unets=true', 'training.cudnn_benchmark=true',
            'training.batch_size=16', 'training.num_epochs=50', 'training.lr=0.001',
            'training.lr_scheduler=step', 'training.lr_step_size=40',
            f'training.seed={seed}', 'misc.resolution=[256,256]',
            f'output.out_dir={root / f"{arm}_seed{seed}"}',
        ])


def brain_loaders(cfg):
    train, val = build_dataloaders(cfg)
    brain = [ds for ds in val.dataset.datasets if ds.anatomy == 'brain']
    assert brain
    assert all(ds.anatomy == 'brain' for ds in train.dataset.datasets)
    val = DataLoader(ConcatDataset(brain), batch_size=cfg.training.batch_size,
                     shuffle=False, num_workers=2, pin_memory=True,
                     persistent_workers=True, collate_fn=val.collate_fn)
    return train, val


def membership(loader):
    return [(str(p), int(s)) for ds in loader.dataset.datasets for p, s in ds.examples]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        name = name.replace('sensetivity_net.model.inner.', 'sensetivity_net.model.')
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_arm(model, arm, compile_body=True):
    if arm == 'moments':
        body = model.sensetivity_net.model
        replace_sensitivity_norms(body, 'moments')
        body.to(memory_format=torch.channels_last)
        if compile_body:
            body.compile(dynamic=False)
        model.sensetivity_net.model = ChannelsLastBody(body)


def preflight(root):
    checks = {}
    for seed in (42, 43):
        cfg = configure(root, seed, 'baseline')
        train, val = brain_loaders(cfg)
        training, validation = membership(train), membership(val)
        assert not ({p for p, _ in training} & {p for p, _ in validation})
        audit = audit_manifest(cfg, train, val)
        write_json(root/'dataset_audit.json', audit)
        order = torch.randperm(len(train.dataset), generator=torch.Generator().manual_seed(seed)).tolist()
        # DataLoader consumes a generator draw for worker base seeds before sampling.
        # Record the real sampler order too, preserving its generator state.
        state = train.generator.get_state()
        actual_order = list(iter(train.sampler))
        train.generator.set_state(state)
        hashes = {}
        for arm in ARMS:
            seed_all(seed)
            model = build_model(build_train_config(cfg))
            set_arm(model, arm, compile_body=False)
            hashes[arm] = state_digest(model)
            del model
        assert hashes['baseline'] == hashes['moments'], hashes
        checks[str(seed)] = {'initial_state_sha256': hashes['baseline'],
            'train_membership_sha256': digest(training), 'val_membership_sha256': digest(validation),
            'sampler_order_sha256': digest(actual_order),
            'train_slices': len(training), 'val_slices': len(validation),
            'train_volumes': len({p for p, _ in training}), 'val_volumes': len({p for p, _ in validation})}
        write_json(root/f'membership_seed{seed}.json', {'train': training, 'val': validation})
        del train, val
    write_json(root/'preflight.json', checks)
    return checks


def audit_manifest(cfg, train, val):
    """Check patient separation and read actual samples before scheduling training."""
    assert cfg.datasets.loader == 'fastmri_manifest'
    manifest = REPO/Path(cfg.datasets.manifest)
    rows = [r for r in csv.DictReader(manifest.open()) if r['cls'] in cfg.datasets.classes]
    selected = {
        'train': [r for r in rows if r['split'] == 'train' and r['tier'] == 'S'],
        'val': [r for r in rows if r['split'] == 'val'],
        'test': [r for r in rows if r['split'] == 'test'],
    }
    assert cfg.datasets.train_tier == 'S'
    patients = {split: {r['patient_id'] for r in records} for split, records in selected.items()}
    assert all(r['patient_id'] and r['anatomy'] == 'brain' for r in rows)
    assert not (patients['train'] & patients['val'])
    assert not (patients['train'] & patients['test'])
    assert not (patients['val'] & patients['test'])
    audit = {'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
             'classes': list(cfg.datasets.classes), 'train_tier': 'S',
             'patient_disjoint': True, 'test_used': False, 'splits': {}}
    for split, loader in (('train', train), ('val', val)):
        examples = membership(loader)
        files = sorted({p for p, _ in examples})
        assert {Path(p).name for p in files} == {r['file'] for r in selected[split]}
        samples = []
        for index in sorted({0, len(loader.dataset)//2, len(loader.dataset)-1}):
            sample = loader.dataset[index]
            k, target = sample['kspace'], sample['target']
            assert k.shape == (cfg.datasets.num_coils, *cfg.misc.resolution, 2)
            assert target.shape == tuple(cfg.misc.resolution)
            assert torch.isfinite(k).all() and torch.isfinite(target).all()
            assert sample['anatomy'] == 'brain' and target.max() > 0
            samples.append({'file': sample['fname'], 'slice': int(sample['slice_id']),
                            'kspace_shape': list(k.shape), 'target_shape': list(target.shape)})
        audit['splits'][split] = {'volumes': len(files), 'slices': len(examples),
            'patients': len(patients[split]), 'samples_checked': samples,
            'files': [{'path': p, 'size': Path(p).stat().st_size} for p in files]}
    return audit


class StudyTrainer(Sequential_Trainer):
    def train_epoch(self, *a, **kw):
        result = super().train_epoch(*a, **kw)
        self.study_train = result
        return result

    def val_epoch(self, *a, **kw):
        result = super().val_epoch(*a, **kw)
        self.study_val = result
        return result

    def save_model(self, is_new_best):
        checkpoint = {'epoch': self.epoch, 'options': self.options,
            'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(), 'best_val_loss': self.best_val_loss,
            'study_arm': self.options.study_arm, 'study_version': 1,
            'loader_rng': self.train_loader.generator.get_state(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}
        for filename in ['last_model.pt'] + (['best_model.pt'] if is_new_best else []):
            temporary = self.exp_dir/(filename+'.tmp')
            torch.save(checkpoint, temporary)
            temporary.replace(self.exp_dir/filename)
        scalar = lambda x: float(x.item()) if hasattr(x, 'item') else float(x)
        tr_loss, tr_time, tr_metrics = self.study_train
        va_loss, _, va_time, va_metrics = self.study_val
        row = {'epoch': self.epoch+1, 'train_loss': scalar(tr_loss), 'val_loss': scalar(va_loss),
               'train_seconds': tr_time, 'val_seconds': va_time,
               'train': {k: scalar(v) for k, v in tr_metrics.items()},
               'val': {k: scalar(v) for k, v in va_metrics.items()}}
        with (self.exp_dir/'epochs.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        write_json(self.exp_dir/'status.json', {'state': 'training', **row})


def train_run(root, seed, arm):
    cfg = configure(root, seed, arm)
    out = root/f'{arm}_seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out/'config.yaml')
    config = build_train_config(cfg)
    config.study_arm = arm
    seed_all(seed)
    torch.backends.cudnn.benchmark = True
    train, val = brain_loaders(cfg)
    trainer = StudyTrainer(config, train, torch.device('cuda'), val_loader=val)
    set_arm(trainer.model, arm)
    expected = json.loads((root/'preflight.json').read_text())[str(seed)]
    assert state_digest(trainer.model) == expected['initial_state_sha256']
    assert digest(membership(train)) == expected['train_membership_sha256']
    assert digest(membership(val)) == expected['val_membership_sha256']
    rng = train.generator.get_state()
    order_hash = digest(list(iter(train.sampler)))
    train.generator.set_state(rng)
    assert order_hash == expected['sampler_order_sha256']
    write_json(out/'matched_initialization.json', {'passed': True, **expected})
    if (out/'last_model.pt').exists():
        ckpt = torch.load(out/'last_model.pt', map_location='cuda', weights_only=False)
        assert ckpt['study_arm'] == arm
        trainer.model.load_state_dict(ckpt['model'])
        trainer.optimizer.load_state_dict(ckpt['optimizer'])
        trainer.scheduler.load_state_dict(ckpt['scheduler'])
        trainer.best_val_loss = ckpt['best_val_loss']
        trainer.start_epoch = ckpt['epoch']+1
        train.generator.set_state(ckpt['loader_rng'].cpu())
        torch.set_rng_state(ckpt['torch_rng'].cpu())
        torch.cuda.set_rng_state(ckpt['cuda_rng'].cpu())
        random.setstate(ckpt['python_rng'])
        np.random.set_state(ckpt['numpy_rng'])
        del ckpt
    trainer()
    write_json(out/'status.json', {'state': 'complete', 'epochs': 50, 'arm': arm, 'seed': seed})


def load_study_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    assert ckpt['epoch'] == 49, 'Primary endpoint requires all 50 epochs'
    model = build_model(ckpt['options']).cuda().eval()
    # Compile the original bodies, then wrap only the sensitivity body, as in training.
    compile_unets(model)
    set_arm(model, ckpt['study_arm'])
    model.load_state_dict(ckpt['model'])
    return model


@torch.no_grad()
def evaluate_seed(root, seed):
    cfg = configure(root, seed, 'baseline')
    _, loader = brain_loaders(cfg)
    all_rows = {}
    for arm in ARMS:
        model = load_study_model(root/f'{arm}_seed{seed}'/'last_model.pt')
        rows = {'volume': [], 'slice': [], 'psnr': [], 'ssim': [], 'apodized_psnr': [], 'apodized_ssim': []}
        for batch in loader:
            k, target = batch['kspace'].cuda(), batch['target'].cuda()
            output = model(target, k)['output']
            mask = metric_mask_from_real_kspace(k, batch['anatomy'])
            metric = MaskedMetricAccumulator(device='cuda')
            metric.update(output, target, mask)
            rows['psnr'].extend(metric.psnr_values)
            rows['ssim'].extend(metric.ssim_values)
            target4 = target.unsqueeze(1) if target.ndim != output.ndim else target
            a, b = _apply_signal_mask(output, target4, mask)
            rows['apodized_psnr'].extend(compute_psnr_torch(a, b).flatten().cpu().tolist())
            rows['apodized_ssim'].extend(compute_ssim_torch(a, b).flatten().cpu().tolist())
            rows['volume'].extend(batch['fname'])
            rows['slice'].extend(batch['slice_id'].tolist())
        all_rows[arm] = rows
        del model
        torch.cuda.empty_cache()
    assert all_rows['baseline']['volume'] == all_rows['moments']['volume']
    assert all_rows['baseline']['slice'] == all_rows['moments']['slice']
    write_json(root/f'paired_seed{seed}.json', all_rows)


def summarize(root):
    rng = np.random.default_rng(20261027)
    summary = {'endpoint': 'epoch 50, brain only, candidate minus baseline', 'seeds': {}}
    seed_volume_deltas = {}
    for seed in (42, 43):
        rows = json.loads((root/f'paired_seed{seed}.json').read_text())
        volumes = np.asarray(rows['baseline']['volume'])
        names = sorted(set(volumes))
        seed_summary = {}
        for metric in ('psnr', 'ssim', 'apodized_psnr', 'apodized_ssim'):
            baseline, candidate = (np.asarray(rows[a][metric]) for a in ARMS)
            delta = candidate-baseline
            vd = np.asarray([delta[volumes == name].mean() for name in names])
            boot = rng.choice(vd, size=(10000, len(vd)), replace=True).mean(1)
            seed_summary[metric] = {'baseline_slice_mean': baseline.mean(),
                'candidate_slice_mean': candidate.mean(), 'delta_slice_mean': delta.mean(),
                'delta_volume_mean': vd.mean(), 'volume_bootstrap_ci95': np.quantile(boot, [.025, .975]).tolist(),
                'worst_volume_delta': vd.min(), 'volumes': len(names)}
            seed_volume_deltas.setdefault(metric, []).append(dict(zip(names, vd.tolist())))
        summary['seeds'][str(seed)] = seed_summary
    summary['seed_averaged_volume_deltas'] = {}
    for metric, pairs in seed_volume_deltas.items():
        assert pairs[0].keys() == pairs[1].keys()
        vd = np.asarray([(pairs[0][v]+pairs[1][v])/2 for v in pairs[0]])
        boot = rng.choice(vd, size=(10000, len(vd)), replace=True).mean(1)
        summary['seed_averaged_volume_deltas'][metric] = {'mean': vd.mean(),
            'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'scope': 'Bootstrap across validation volumes, conditional on these two training seeds'}
    for seed in (42, 43):
        for arm in ARMS:
            epochs = [json.loads(l) for l in (root/f'{arm}_seed{seed}'/'epochs.jsonl').read_text().splitlines()]
            summary.setdefault('training_seconds', {})[f'{arm}_seed{seed}'] = {
                'total': sum(e['train_seconds'] for e in epochs),
                'median_after_epoch1': float(np.median([e['train_seconds'] for e in epochs if e['epoch'] > 1]))}
    write_json(root/'comparison.json', summary)
    lines = ['# Brain-only, 12-cascade normalization study', '',
             'Completed 50 epochs for two matched seeds. Primary endpoint uses final checkpoints; no best-checkpoint selection.', '',
             '| Seed | Baseline masked PSNR | Moments masked PSNR | Delta dB | Volume-paired 95% CI |',
             '|---|---:|---:|---:|---|']
    for seed, results in summary['seeds'].items():
        s = results['psnr']
        lines.append(f"| {seed} | {s['baseline_slice_mean']:.4f} | {s['candidate_slice_mean']:.4f} | "
                     f"{s['delta_slice_mean']:+.4f} | {s['volume_bootstrap_ci95']} |")
    lines += ['', 'The displayed means/delta average slices; confidence intervals average paired volume means. '
              'See comparison.json for both aggregation units and SSIM. Intervals do not capture uncertainty across arbitrary training seeds.',
              '', 'Both arms use full-resolution 256x256 sensitivity estimation. Only normalization/layout differs; FFT shifts are unchanged.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def supervise(root):
    script = str(REPO/'scripts/diagnostics/brain_sensnorm_study.py')
    def wait_for_free_gpus(indices):
        while True:
            mapping = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                              '--format=csv,noheader'], text=True)
            wanted = {line.split(',')[1].strip() for line in mapping.splitlines()
                      if int(line.split(',')[0]) in indices}
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                           '--format=csv,noheader'], text=True)
            busy = [line for line in apps.splitlines() if line.split(',')[0].strip() in wanted]
            if not busy:
                return
            write_json(root/'status.json', {'state': 'waiting_for_free_gpus', 'gpus': indices, 'processes': busy})
            time.sleep(30)
    for seed in (42, 43):
        wait_for_free_gpus([2, 3])
        processes = []
        gpu_order = (2, 3) if seed == 42 else (3, 2)
        for arm, gpu in zip(ARMS, gpu_order):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CC='gcc', CXX='g++')
            log = (root/f'{arm}_seed{seed}.log').open('a')
            proc = subprocess.Popen([sys.executable, script, 'train', '--root', str(root),
                                     '--seed', str(seed), '--arm', arm], env=env, stdout=log,
                                    stderr=subprocess.STDOUT, cwd=REPO)
            processes.append((arm, gpu, proc, log))
        write_json(root/'status.json', {'state': 'training', 'seed': seed,
            'jobs': [{'arm': a, 'gpu': g, 'pid': p.pid} for a, g, p, _ in processes]})
        while any(p.poll() is None for _, _, p, _ in processes):
            failed = [(a, p.returncode) for a, _, p, _ in processes if p.poll() not in (None, 0)]
            if failed:
                for _, _, p, _ in processes:
                    if p.poll() is None:
                        p.terminate()
                break
            time.sleep(10)
        codes = [(a, p.wait()) for a, _, p, _ in processes]
        for _, _, _, log in processes:
            log.close()
        if any(code for _, code in codes):
            write_json(root/'status.json', {'state': 'failed', 'seed': seed, 'returncodes': codes})
            raise RuntimeError(codes)
        write_json(root/'status.json', {'state': 'evaluating', 'seed': seed})
        wait_for_free_gpus([2])
        with (root/f'evaluation_seed{seed}.log').open('a') as log:
            subprocess.run([sys.executable, script, 'evaluate', '--root', str(root), '--seed', str(seed)],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='2', CC='gcc', CXX='g++'),
                stdout=log, stderr=subprocess.STDOUT, cwd=REPO, check=True)
    summarize(root)
    write_json(root/'status.json', {'state': 'complete', 'results': str(root/'RESULTS.md')})


def prepare(root):
    """Freeze code/data selection and run CPU preflight without starting training."""
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root/'code'
    if not snapshot.exists():
        for directory in ('activemri', 'configs', 'scripts'):
            shutil.copytree(REPO/directory, snapshot/directory,
                            ignore=shutil.ignore_patterns('__pycache__'), symlinks=True)
        shutil.copy2(REPO/'train.py', snapshot/'train.py')
        shutil.copy2(MASK, root/'mask.pt')
        write_json(root/'source_sha256.json', {
            str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(snapshot.rglob('*')) if p.is_file() and not p.is_symlink()
            and '__pycache__' not in p.parts})
    frozen = snapshot/'scripts/diagnostics/brain_sensnorm_study.py'
    try:
        subprocess.run([sys.executable, str(frozen), 'preflight', '--root', str(root)], check=True, cwd=snapshot)
    except Exception as error:
        write_json(root/'status.json', {'state': 'failed', 'phase': 'preflight',
                                      'error': str(error), 'failed_unix': time.time()})
        raise
    write_json(root/'study.json', {'seeds': [42, 43], 'arms': ARMS, 'epochs': 50,
        'cascades': 12, 'resolution': [256, 256], 'training_anatomy': 'brain',
        'primary_endpoint': 'Final epoch masked PSNR/SSIM with volume-paired differences',
        'baseline': 'Current compiled BF16 InstanceNorm', 'candidate': 'Full-resolution FP32 spatial moments + channels-last sensitivity body',
        'mask_sha256': hashlib.sha256((root/'mask.pt').read_bytes()).hexdigest(),
        'dataset_audit': str(root/'dataset_audit.json'),
        'prepared_unix': time.time(), 'snapshot': str(snapshot)})
    return frozen, snapshot


def launch(root):
    if (root/'supervisor.pid').exists():
        pid = int((root/'supervisor.pid').read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f'Supervisor {pid} already exists')
    frozen, snapshot = prepare(root)
    write_json(root/'status.json', {'state': 'starting', 'started_unix': time.time()})
    with (root/'supervisor.log').open('a') as log:
        proc = subprocess.Popen([sys.executable, str(frozen), 'supervise', '--root', str(root)],
            stdout=log, stderr=subprocess.STDOUT, cwd=snapshot, start_new_session=True)
    (root/'supervisor.pid').write_text(str(proc.pid)+'\n')
    print(f'Supervisor PID {proc.pid}; status {root / "status.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'launch', 'preflight', 'supervise', 'train', 'evaluate', 'summarize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arm', choices=ARMS, default='baseline')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'train': train_run(root, args.seed, args.arm)
    elif args.action == 'evaluate': evaluate_seed(root, args.seed)
    else: globals()[args.action](root)
