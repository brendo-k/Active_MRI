"""Pre-launch pairing check: built as train.py builds them at one seed, every arm sees the same first train batch, the
same initial weights and the same mask draw, and its loss and gradients are finite.

    CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python reports/loss_domain/check_pairing.py --seed 42
"""
import argparse
import hashlib
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from activemri.training.builders import build_dataloaders, build_train_config  # noqa: E402
from activemri.training.trainer import Sequential_Trainer  # noqa: E402
from train import seed_everything  # noqa: E402

CACHE = '/export/scratch/bkadota/fastmri_complex_espirit'
SHARED = ['recon=varnet', 'datasets=fastmri_local_v1_flair_knee', 'misc.resolution=[320,320]', 'sampler.accelerations=[8.0]',
          'sampler.budget_over=acquired', 'sampler.sampler_type=loupe_2d', 'sampler.loupe_vd_init_power=4',
          'sampler.anatomy_conditioning=false', 'sampler.coil_combine=sense', 'recon.csme=false', 'training.num_step=1',
          'training.batch_size=16', 'training.num_epochs=1', 'training.lr_step_size=90', 'training.compile_unets=false']
ARMS = {
    'rss': ['loss.loss_type=l1'],
    'complex': ['loss.loss_type=complex_l1', f'datasets.complex_cache_dir={CACHE}'],
    'kspace_sense': ['loss.loss_type=kspace_sense_l1', f'datasets.complex_cache_dir={CACHE}'],
    'kspace_rss': ['loss.loss_type=kspace_rss_l1'],
    'kspace_rss_M': ['loss.loss_type=kspace_rss_l1', 'loss.support_mask=true', f'datasets.complex_cache_dir={CACHE}'],
}


def digest(tensors):
    hasher = hashlib.sha1()
    for tensor in tensors:
        hasher.update(tensor.detach().float().cpu().numpy().tobytes())
    return hasher.hexdigest()[:12]


def check(arm, seed, out_dir):
    with initialize_config_dir(config_dir=str(REPO / 'configs'), version_base=None):
        cfg = compose('config', overrides=SHARED + ARMS[arm] + [f'training.seed={seed}', f'output.out_dir={out_dir / arm}'])
    config = build_train_config(cfg)
    seed_everything(seed)
    train_loader, val_loader = build_dataloaders(cfg)
    trainer = Sequential_Trainer(config, train_loader, torch.device('cuda'), val_loader=val_loader)
    trainer.model.train()
    initial = digest(p for p in trainer.model.parameters())
    batch = next(iter(train_loader))
    kspace = trainer._load_kspace(batch)
    target = trainer._target(batch, kspace)
    pred_dict = trainer._train_forward(target, kspace, batch)
    meta = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0,
            'support_mask': config.loss_support_mask}
    loss, _ = trainer.model.loss(pred_dict, trainer._target_dict(batch, target, kspace), meta, config.loss_type)
    trainer.optimizer.zero_grad()
    loss.backward()
    grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
    sampler_grad = trainer.model.sampler.mask_net.mask.grad
    result = {
        'fnames': list(zip(batch['fname'], batch['slice_id'].tolist())),
        'init': initial,
        'mask': digest([pred_dict['mask']]),
        'loss': loss.item(),
        'finite_grads': all(torch.isfinite(g).all().item() for g in grads),
        'sampler_grad_norm': sampler_grad.norm().item() if sampler_grad is not None else 0.0,
        'peak_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
    }
    del trainer, pred_dict, loss, grads
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, default=Path('/tmp/loss_domain_pairing'))
    args = parser.parse_args()
    results = {arm: check(arm, args.seed, args.out) for arm in ARMS}
    reference = results['rss']
    ok = True
    for arm, result in results.items():
        same = {key: result[key] == reference[key] for key in ('fnames', 'init', 'mask')}
        ok &= all(same.values()) and result['finite_grads'] and torch.isfinite(torch.tensor(result['loss'])).item()
        print(f"{arm:13s} loss {result['loss']:.5g}  same batch/init/mask as rss {same}  finite grads "
              f"{result['finite_grads']}  sampler grad norm {result['sampler_grad_norm']:.3g}  peak {result['peak_gb']:.1f} GB")
    print('first batch:', reference['fnames'][:4], '...')
    print('PAIRING OK' if ok else 'PAIRING FAILED')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
