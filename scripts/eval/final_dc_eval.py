"""Score checkpoints with a final data-consistency step on the scored output (reports/final_dc/proposal.md).

The step runs after the model's forward pass, on its own `recon_complex` x, `sense_maps` S and `mask` M:

    K = F(S x) E;  K_l = K + l M E (k0 - K);  output = sqrt(sum_c |F^-1 K_l|^2 + RSS_EPSILON)

l = 0 is the model's own output (`_coil_rss`). Every l comes from one forward pass. Each checkpoint is scored with the code it
trained with (--code-root, imported in place of this tree's `activemri`) and with that study's val loader (--loader), from
the code root as working directory, because both loaders resolve the manifest relative to it. Two seeded mask draws
(global RNG seeded before each pass), averaged per slice.

Per run it writes OUT/<tag>_<run>.npz: for every l, per-slice PSNR_Masked, SSIM_Masked, NMSE and output/target mean inside the
apodized metric mask; the oracle gain+offset PSNR at l in FIT_LAMBDAS; and at l = 0 the relative error at the sampled points,
the maps' own floor there, and the share of the coil k-space error inside E that sits at the sampled points.

    cd ../run-ml100-91932da && CUDA_VISIBLE_DEVICES=0 python ../exp-final_dc/scripts/eval/final_dc_eval.py \
        --code-root . --loader ml100 --tag ml100 --out ../exp-final_dc/reports/final_dc/scores \
        --runs ~/data/seqmri_checkpoints/muon_layernorm_100ep/control_s42
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FIT_LAMBDAS = (0.0, 1.0)
MASK_DRAW_SEEDS = (1234, 4321)
PHASE_LOWPASS_SIGMA = 8.0  # k-space px; fixed before the full run, not tuned on val
ML100_CONFIG = Path('~/data/seqmri_checkpoints/muon_layernorm_100ep/control_s42/.hydra/config.yaml').expanduser()
LOSS_DOMAIN_ROOTS = {'local_batch2': '/export/scratch/bkadota/fastmri_knee/batch2',
                     'hpc': '/export/scratch/bkadota/fastmri_raw/{anatomy}'}  # scripts/eval/eval_loss_domain.py on exp/loss_domain


def centred_gaussian(shape, sigma, device):
    rows, cols = (torch.arange(n, device=device) - n // 2 for n in shape)
    return torch.exp(-(rows[:, None] ** 2 + cols[None, :] ** 2) / (2 * sigma ** 2))


def per_slice(x):
    return x.flatten(1)


def masked_mean(x, m):
    return per_slice(x * m).sum(1) / per_slice(m).sum(1).clamp(min=1.0)


def masked_psnr(output, target, m):
    mse = per_slice((target - output) ** 2 * m).sum(1) / per_slice(m).sum(1).clamp(min=1.0)
    return 10 * torch.log10(per_slice(target).amax(1) ** 2 / mse)


def gain_offset_psnr(output, target, m):
    """PSNR after the least-squares fit target ~ a output + b inside m, per slice; an oracle bound, as intensity_fit.py."""
    mo, mt = masked_mean(output, m), masked_mean(target, m)
    do, dt = output - mo[:, None, None, None], target - mt[:, None, None, None]
    a = masked_mean(do * dt, m) / masked_mean(do * do, m)
    b = mt - a * mo
    return masked_psnr(a[:, None, None, None] * output + b[:, None, None, None], target, m)


def relative_norm(x, reference):
    return per_slice(x.abs() ** 2).sum(1).sqrt() / per_slice(reference.abs() ** 2).sum(1).sqrt().clamp(min=1e-30)


def load_loader(kind, code):
    if kind == 'ml100':
        from omegaconf import OmegaConf
        _, val_loader = code.build_dataloaders(OmegaConf.load(ML100_CONFIG))
        return val_loader
    _, val_loader = code.build_manifest_dataloaders(
        manifest='configs/datasets/fastmri_local_v1/manifest.csv', roots=LOSS_DOMAIN_ROOTS, classes=['FLAIR', 'knee'],
        image_shape=[320, 320], batch_size=16, num_coils=20, num_workers=8)
    return val_loader


def load_model(kind, run_dir, code, device):
    """As the study's own eval loaded it: varnet_train_speed_eval.py (ml100) or eval_loss_domain.py (loss_domain)."""
    checkpoint = torch.load(run_dir / 'last_model.pt', map_location='cpu', weights_only=False)
    model = code.build_model(checkpoint['options']).to(device)
    if kind == 'ml100':
        model.load_state_dict(checkpoint['model'])
        model.eval()
        if hasattr(model, 'store_visuals'):
            model.store_visuals = False
    else:
        model.load_state_dict({k.removeprefix('module.'): v for k, v in checkpoint['model'].items()})
        model.eval()
        model.sampler.eval_generator = None
    return model, checkpoint


@torch.no_grad()
def score(kind, run_dir, val_loader, code, device, max_batches):
    model, checkpoint = load_model(kind, run_dir, code, device)
    options = checkpoint['options']
    classes = list(getattr(options, 'label_classes', []) or []) if kind == 'ml100' else []
    keys = [f'{metric}_l{l:g}' for l in LAMBDAS for metric in ('psnr', 'ssim', 'nmse', 'mean_ratio')]
    keys += [f'fit_psnr_l{l:g}' for l in FIT_LAMBDAS] + ['dc_error', 'maps_floor', 'error_share_sampled', 'sampled_fraction']
    keys += ['phase_offset', 'dc_error_aligned'] + [f'{m}_aligned_l{l:g}' for l in FIT_LAMBDAS for m in ('psnr', 'ssim')]
    keys += ['phase_estimate_error', 'dc_error_estaligned'] + [f'{m}_estaligned_l{l:g}' for l in FIT_LAMBDAS
                                                                for m in ('psnr', 'ssim', 'mean_ratio')]
    draws = {k: [[] for _ in MASK_DRAW_SEEDS] for k in keys}
    conditions, volumes, own_output_gap = [], [], 0.0
    for draw, seed in enumerate(MASK_DRAW_SEEDS):
        torch.manual_seed(seed)
        for batch in itertools.islice(val_loader, max_batches):
            kspace_real = batch['kspace'].to(device)
            target = batch['target'].to(device)
            target = target if target.dim() == 4 else target.unsqueeze(1)
            anatomies = list(batch['anatomy'])
            labels = torch.tensor([classes.index(c) for c in batch[options.label_key]], device=device) if classes else None
            pred = model(target, kspace_real, anatomy_labels=labels) if kind == 'ml100' else model(target, kspace_real)
            own = pred['output'] if pred['output'].dim() == 4 else pred['output'].unsqueeze(1)

            k0 = torch.view_as_complex(kspace_real.contiguous())
            maps, x, mask = pred['sense_maps'], pred['recon_complex'], pred['mask']
            extent = model._kspace_extent(kspace_real)
            sampled = mask * extent
            coil_kspace = code.fft(x * maps) * extent
            residual = sampled * (k0 - coil_kspace)
            metric = code.metric_support_mask(k0, anatomies).float()

            for l in LAMBDAS:
                output = (code.ifft(coil_kspace + l * residual).abs().pow(2).sum(dim=1, keepdim=True) + code.RSS_EPSILON).sqrt()
                if l == 0:
                    own_output_gap = max(own_output_gap, float((output - own.float()).abs().max()))
                draws[f'psnr_l{l:g}'][draw].append(masked_psnr(output, target, metric))
                draws[f'ssim_l{l:g}'][draw].append(code.masked_ssim(output, target, metric))
                draws[f'nmse_l{l:g}'][draw].append(per_slice((output - target) ** 2 * metric).sum(1)
                                                   / per_slice(target ** 2 * metric).sum(1))
                draws[f'mean_ratio_l{l:g}'][draw].append(masked_mean(output, metric) / masked_mean(target, metric))
                if l in FIT_LAMBDAS:
                    draws[f'fit_psnr_l{l:g}'][draw].append(gain_offset_psnr(output, target, metric))

            coil_images = code.ifft(k0)
            fully_sampled = (coil_images * maps.conj()).sum(dim=1, keepdim=True)
            maps_floor = sampled * (k0 - code.fft(fully_sampled * maps) * extent)

            # Post-freeze diagnostic: the per-pixel common phase of S x against the measured coil images, which an RSS loss
            # leaves free, and the same step after an oracle rotation of x onto the data's phase.
            phase_offset = torch.angle((coil_images.conj() * maps * x).sum(dim=1, keepdim=True))
            tissue = target * metric
            draws['phase_offset'][draw].append(per_slice(phase_offset.abs() * tissue).sum(1) / per_slice(tissue).sum(1))
            aligned_kspace = code.fft(x * torch.exp(-1j * phase_offset) * maps) * extent
            aligned_residual = sampled * (k0 - aligned_kspace)
            draws['dc_error_aligned'][draw].append(relative_norm(aligned_residual, sampled * k0))
            for l in FIT_LAMBDAS:
                output = (code.ifft(aligned_kspace + l * aligned_residual).abs().pow(2).sum(dim=1, keepdim=True)
                          + code.RSS_EPSILON).sqrt()
                draws[f'psnr_aligned_l{l:g}'][draw].append(masked_psnr(output, target, metric))
                draws[f'ssim_aligned_l{l:g}'][draw].append(code.masked_ssim(output, target, metric))

            # The same rotation estimated from the sampled data alone: both k-spaces through the same samples and low-pass.
            lowpass = sampled * centred_gaussian(k0.shape[-2:], PHASE_LOWPASS_SIGMA, device)
            estimate = torch.angle((code.ifft(lowpass * k0).conj() * code.ifft(lowpass * coil_kspace)).sum(dim=1, keepdim=True))
            draws['phase_estimate_error'][draw].append(
                per_slice(torch.angle(torch.exp(1j * (phase_offset - estimate))).abs() * tissue).sum(1) / per_slice(tissue).sum(1))
            estimated_kspace = code.fft(x * torch.exp(-1j * estimate) * maps) * extent
            estimated_residual = sampled * (k0 - estimated_kspace)
            draws['dc_error_estaligned'][draw].append(relative_norm(estimated_residual, sampled * k0))
            for l in FIT_LAMBDAS:
                output = (code.ifft(estimated_kspace + l * estimated_residual).abs().pow(2).sum(dim=1, keepdim=True)
                          + code.RSS_EPSILON).sqrt()
                draws[f'psnr_estaligned_l{l:g}'][draw].append(masked_psnr(output, target, metric))
                draws[f'ssim_estaligned_l{l:g}'][draw].append(code.masked_ssim(output, target, metric))
                draws[f'mean_ratio_estaligned_l{l:g}'][draw].append(masked_mean(output, metric) / masked_mean(target, metric))

            draws['dc_error'][draw].append(relative_norm(residual, sampled * k0))
            draws['maps_floor'][draw].append(relative_norm(maps_floor, sampled * k0))
            draws['error_share_sampled'][draw].append(relative_norm(residual, extent * (k0 - coil_kspace)) ** 2)
            draws['sampled_fraction'][draw].append(per_slice(sampled).sum(1) / per_slice(extent).sum(1))
            if draw == 0:
                conditions += list(batch['condition'])
                volumes += [f"{c}/{f}" for c, f in zip(batch['condition'], batch['fname'])]
    out = {k: torch.stack([torch.cat(d).double() for d in v]).mean(0).cpu().numpy() for k, v in draws.items()}
    return {**out, 'condition': np.array(conditions), 'volume': np.array(volumes), 'epochs': checkpoint['epoch'] + 1,
            'own_output_max_abs_gap': own_output_gap, 'lambdas': np.array(LAMBDAS)}


def import_code(code_root):
    sys.path.insert(0, str(code_root.resolve()))
    from types import SimpleNamespace
    from activemri.datasets.fastmri_data import build_manifest_dataloaders
    from activemri.models.sequential_sampling import RSS_EPSILON
    from activemri.training.trainer import build_model
    from activemri.utils.image_processing import fft_2d_img, ifft_2d_img
    from activemri.utils.metric_mask import metric_support_mask
    from activemri.utils.metrics import masked_ssim
    try:
        from activemri.training.builders import build_dataloaders
    except ImportError:
        build_dataloaders = None
    return SimpleNamespace(build_manifest_dataloaders=build_manifest_dataloaders, build_dataloaders=build_dataloaders,
                           RSS_EPSILON=RSS_EPSILON, build_model=build_model, fft=fft_2d_img, ifft=ifft_2d_img,
                           metric_support_mask=metric_support_mask, masked_ssim=masked_ssim)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--code-root', type=Path, required=True, help='the tree whose activemri the runs trained with')
    parser.add_argument('--loader', choices=('ml100', 'loss_domain'), required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True, help='run directories holding last_model.pt')
    parser.add_argument('--tag', required=True, help='prefix of the output files, e.g. ml100 or loss_domain')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-batches', type=int, help='score only this many val batches per draw (to test the script)')
    args = parser.parse_args()
    if Path.cwd().resolve() != args.code_root.resolve():
        raise SystemExit(f'run from the code root ({args.code_root}): its loader resolves the manifest against the working directory')
    code = import_code(args.code_root)
    imported_from = Path(sys.modules['activemri'].__file__).resolve().parents[1]
    if imported_from != args.code_root.resolve():
        raise SystemExit(f'activemri was imported from {imported_from}, not the code root {args.code_root.resolve()}')
    print(f'activemri from {imported_from}', flush=True)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    val_loader = load_loader(args.loader, code)
    for run_dir in args.runs:
        run_dir = run_dir.expanduser()
        path = args.out / f'{args.tag}_{run_dir.name}.npz'
        if path.exists() and args.max_batches is None:
            print(f'{path.name} exists, skipped', flush=True)
            continue
        print(f'scoring {run_dir}', flush=True)
        result = score(args.loader, run_dir, val_loader, code, device, args.max_batches)
        np.savez(path, **result)
        print(f"{run_dir.name}: ep {result['epochs']}  own-output gap {result['own_output_max_abs_gap']:.2e}  " + '  '.join(
            f"l{l:g} {result[f'psnr_l{l:g}'].mean():.3f}/{result[f'ssim_l{l:g}'].mean():.4f}" for l in LAMBDAS)
            + f"  dc_error {result['dc_error'].mean():.3f}  maps_floor {result['maps_floor'].mean():.3f}"
            + f"  error_share_sampled {result['error_share_sampled'].mean():.3f}  phase_offset {result['phase_offset'].mean():.3f} rad"
            + f"  aligned l0/l1 {result['psnr_aligned_l0'].mean():.3f}/{result['psnr_aligned_l1'].mean():.3f}"
            + f"  dc_error_aligned {result['dc_error_aligned'].mean():.3f}"
            + f"  estimated l0/l1 {result['psnr_estaligned_l0'].mean():.3f}/{result['psnr_estaligned_l1'].mean():.3f}"
            + f"  phase_estimate_error {result['phase_estimate_error'].mean():.3f} rad", flush=True)


if __name__ == '__main__':
    main()
