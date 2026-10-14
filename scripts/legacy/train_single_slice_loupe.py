"""Overfit a zero-step LOUPE sampler to a single slice.

A zero-step run has no adaptive feedback: the sampler sees no observed k-space,
so the learned mask is a pure function of the one slice it is trained on. Running
this on a brain slice, a knee slice, and a second brain slice shows how much of
the learned pattern is anatomy-specific and how much is slice-specific.

No validation split — the point is to fit one example as well as possible and
read off the mask.

    python scripts/legacy/train_single_slice_loupe.py --anatomy brain --slice-id 8 \
        --sampler-type loupe_2d --out-dir runs/single_slice/brain_s8_loupe2d
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import _path  # noqa: F401
from activemri.config import TrainConfig
from activemri.datasets.fastmri_data import FastMRIDataset, _resolve_anatomy_dirs
from activemri.reconstructors.config import UNetReconConfig, VarNetReconConfig, CGSenseReconConfig
from activemri.training.trainer import build_model
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch

RECON_CONFIGS = {
    'unet': UNetReconConfig,
    'varnet': VarNetReconConfig,
    'cg_sense': CGSenseReconConfig,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--anatomy', required=True, choices=['brain', 'knee', 'breast', 'prostate'])
    parser.add_argument('--contrast', default='T2')
    parser.add_argument('--file-name', default=None, help='h5 file name; defaults to the first file of the anatomy')
    parser.add_argument('--slice-id', type=int, required=True)
    parser.add_argument('--num-coils', type=int, default=16)
    parser.add_argument('--resolution', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--sampler-type', default='loupe_2d', choices=['loupe_2d', 'loupe_1d'])
    parser.add_argument('--acceleration', type=float, default=4.0)
    parser.add_argument('--preselect-num', type=int, default=10)
    parser.add_argument('--slope', type=float, default=10.0,
                        help='sigmoid slope on the mask logits. The init is slope-normalized, so a '
                             'larger slope leaves the starting probabilities alone but scales up the '
                             'gradient reaching them.')
    parser.add_argument('--recon-type', default='unet', choices=list(RECON_CONFIGS))
    parser.add_argument('--loss-type', default='l1', choices=['l1', 'ssim', 'nmse', 'psnr'])
    parser.add_argument('--iters', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mask-lr', type=float, default=None,
                        help='learning rate for the LOUPE mask parameter; defaults to --lr. '
                             'The mask needs a much larger step than the reconstructor to converge.')
    parser.add_argument('--mask-init', default='uniform', choices=['uniform', 'constant'],
                        help="'constant' starts every location at p=0.5 so the learned pattern "
                             'cannot inherit structure from the initialization.')
    parser.add_argument('--mask-train-mode', default='straight_through',
                        choices=['straight_through', 'reinforce'],
                        help="'reinforce' replaces the straight-through binarization with exact-"
                             'budget Plackett-Luce sampling, trained by REINFORCE -- see '
                             'activemri/samplers/sampler.py. Requires --reinforce-batch >= 2: since '
                             'the LOUPE mask ignores its input, each Monte Carlo draw needs its own '
                             'batch slot, so this script replicates the one slice across the batch.')
    parser.add_argument('--reinforce-batch', type=int, default=8,
                        help='Monte Carlo mask draws per step under --mask-train-mode reinforce '
                             '(the single slice is replicated this many times); also the REINFORCE '
                             'baseline sample size, so bigger reduces gradient variance.')
    parser.add_argument('--reinforce-weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out-dir', type=Path, required=True)
    return parser.parse_args()


def load_single_slice(args):
    train_dir, _ = _resolve_anatomy_dirs(args.data_root, args.anatomy, args.contrast)

    file_names = [args.file_name] if args.file_name else None
    if file_names is None:
        probe = FastMRIDataset(
            root=train_dir, anatomy=args.anatomy, image_shape=args.resolution,
            num_coils=args.num_coils, min_energy_fraction=0.0, slice_ids=[0],
        )
        file_names = [probe.examples[0][0].name]

    dataset = FastMRIDataset(
        root=train_dir,
        anatomy=args.anatomy,
        image_shape=args.resolution,
        num_coils=args.num_coils,
        min_energy_fraction=0.0,
        file_names=file_names,
        slice_ids=[args.slice_id],
    )
    if len(dataset) != 1:
        raise RuntimeError(f"Expected exactly one example, got {len(dataset)}")

    sample = dataset[0]
    return sample, file_names[0]


def build_recon_config(recon_type):
    if recon_type == 'unet':
        return UNetReconConfig(in_chans=2, out_chans=2)
    return RECON_CONFIGS[recon_type]()


def build_config(args):
    return TrainConfig(
        model='SequentialSampling',
        num_step=1,
        fixed_input=True,  # zero-step: the sampler is not conditioned on observations
        batch_size=1,
        num_epochs=args.iters,
        lr=args.lr,
        loss_type=args.loss_type,
        anatomy_classes=[],
        accelerations=[args.acceleration],
        sampler_type=args.sampler_type,
        slope=args.slope,
        preselect_num=args.preselect_num,
        mask_train_mode=args.mask_train_mode,
        reinforce_weight=args.reinforce_weight,
        recon=build_recon_config(args.recon_type),
        resolution=list(args.resolution),
        out_dir=[args.out_dir],
        device=args.device,
        seed=args.seed,
    )


def flatten_mask_init(model, jitter=1e-4):
    """Start every k-space location at p=0.5, so structure can only come from data."""
    mask = model.sampler.mask_net.mask
    with torch.no_grad():
        mask.copy_(torch.empty_like(mask).uniform_(-jitter, jitter))


def build_optimizer(model, args):
    """Give the mask its own learning rate; it converges far slower than the reconstructor."""
    mask_params = set(model.sampler.mask_net.parameters())
    rest = [p for p in model.parameters() if p not in mask_params]
    mask_lr = args.mask_lr if args.mask_lr is not None else args.lr
    return torch.optim.Adam(
        [{'params': rest, 'lr': args.lr},
         {'params': list(mask_params), 'lr': mask_lr}],
        lr=args.lr,
    )


def learned_probabilities(model, options):
    """The LOUPE free parameter as probabilities, before sparsity rescaling."""
    logits = model.sampler.mask_net.mask.detach()
    prob = torch.sigmoid(options.slope * logits)
    if options.sampler_type == 'loupe_1d':
        return prob.view(-1).cpu().numpy()
    return prob.view(options.resolution[0], options.resolution[1]).cpu().numpy()


def save_outputs(args, options, model, sample, pred_dict, history, file_name):
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prob = learned_probabilities(model, options)
    np.save(out_dir / 'learned_prob.npy', prob)

    vis = model.data_for_vis[0]
    rescaled = vis['rescaled_mask'].cpu().numpy()
    binary = vis['binarized_mask'].cpu().numpy()
    np.save(out_dir / 'rescaled_mask.npy', rescaled)
    np.save(out_dir / 'binary_mask.npy', binary)

    target = sample['target'].cpu().numpy()
    recon = pred_dict['output'][0, 0].detach().cpu().numpy()
    np.save(out_dir / 'target.npy', target)
    np.save(out_dir / 'recon.npy', recon)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
    ax[0].imshow(target, cmap='gray')
    ax[0].set_title(f'{args.anatomy} slice {args.slice_id}')
    ax[1].imshow(recon, cmap='gray')
    ax[1].set_title('Reconstruction')
    # Masks live in the same DC-centered k-space convention as the data, so they
    # are plotted as-is.
    if prob.ndim == 1:
        ax[2].plot(prob)
        ax[2].set_title('Learned line probability')
        ax[2].set_xlabel('phase encode')
        ax[3].imshow(binary, cmap='gray', aspect='auto')
    else:
        ax[2].imshow(prob, cmap='inferno')
        ax[2].set_title('Learned probability')
        ax[3].imshow(binary, cmap='gray')
    ax[3].set_title('Binary mask')
    for a in (ax[0], ax[1], ax[3]):
        a.axis('off')
    fig.suptitle(f"{args.anatomy} {file_name} slice {args.slice_id} — {args.sampler_type} R={args.acceleration:g}")
    fig.tight_layout()
    fig.savefig(out_dir / 'summary.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history['loss'])
    ax.set_xlabel('iteration')
    ax.set_ylabel(f'{args.loss_type} loss')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(out_dir / 'loss.png', dpi=150)
    plt.close(fig)

    metadata = {
        'anatomy': args.anatomy,
        'file_name': file_name,
        'slice_id': args.slice_id,
        'sampler_type': args.sampler_type,
        'slope': args.slope,
        'acceleration': args.acceleration,
        'preselect_num': args.preselect_num,
        'recon_type': args.recon_type,
        'loss_type': args.loss_type,
        'iters': args.iters,
        'lr': args.lr,
        'mask_lr': args.mask_lr if args.mask_lr is not None else args.lr,
        'mask_init': args.mask_init,
        'seed': args.seed,
        'resolution': list(args.resolution),
        'final_loss': history['loss'][-1],
        'final_psnr': history['psnr'][-1],
        'final_ssim': history['ssim'][-1],
        'mask_sparsity': float(binary.mean()),
    }
    (out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sample, file_name = load_single_slice(args)
    device = torch.device(args.device)
    kspace = sample['kspace'].unsqueeze(0).to(device)
    target = sample['target'].unsqueeze(0).to(device)

    if args.mask_train_mode == 'reinforce':
        # The LOUPE mask ignores its input, so replicating the one slice across
        # the batch gives `reinforce_batch` independent Monte Carlo draws of the
        # same (input-independent) policy per step -- exactly what the REINFORCE
        # loss term in SequentialSampling.loss needs a batch of >= 2 for.
        if args.reinforce_batch < 2:
            raise ValueError('--reinforce-batch must be >= 2 under --mask-train-mode reinforce')
        # repeat (not expand): the pipeline has enough views/in-place-ish ops
        # downstream that a stride-0 batch dim risks silent aliasing bugs.
        kspace = kspace.repeat(args.reinforce_batch, *([1] * (kspace.dim() - 1)))
        target = target.repeat(args.reinforce_batch, *([1] * (target.dim() - 1)))

    options = build_config(args)
    model = build_model(options).to(device)
    if args.mask_init == 'constant':
        flatten_mask_init(model)
    optimizer = build_optimizer(model, args)

    meta = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0}
    target_dict = {'target': target, 'kspace': kspace}

    history = {'loss': [], 'psnr': [], 'ssim': []}
    model.train()
    for step in range(args.iters):
        pred_dict = model(target, kspace)
        loss, _ = model.loss(pred_dict, target_dict, meta, args.loss_type)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        with torch.no_grad():
            output = pred_dict['output'].detach()
            target_4d = target if target.dim() == 4 else target.unsqueeze(1)
            psnr = compute_psnr_torch(output, target_4d).mean().item()
            ssim = compute_ssim_torch(output, target_4d).mean().item()

        history['loss'].append(loss.item())
        history['psnr'].append(psnr)
        history['ssim'].append(ssim)

        if step % 100 == 0 or step == args.iters - 1:
            print(f'[{args.anatomy} s{args.slice_id} {args.sampler_type}] '
                  f'iter {step:5d}/{args.iters} loss={loss.item():.5f} psnr={psnr:.2f} ssim={ssim:.4f}',
                  flush=True)

    model.eval()
    with torch.no_grad():
        pred_dict = model(target, kspace)

    metadata = save_outputs(args, options, model, sample, pred_dict, history, file_name)
    np.savez(args.out_dir / 'history.npz', **{k: np.array(v) for k, v in history.items()})
    torch.save({'model': model.state_dict(), 'options': options}, args.out_dir / 'model.pt')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
