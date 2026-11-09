"""Pre-launch QC for the complex_maps arm (amendment 2026-11-03), from the round-2 rss seed-42 checkpoint on every 12th
val slice (162; eval mode, mask draw 1234).

- Ceiling: the complex_maps target x_L = sum_c conj(S_c) F^-1(k_c E) through the rss arm's learned maps, scored like an
  output (RSS of F^-1(F(S x_L) E)) against the RSS target t. A perfect complex_maps fit scores this. The rss arm's own
  output on the same slices is printed beside it.
- Phase order of angle(x_L conj x_E), in tissue (metric mask and |x_E| > 0.3 max), weighted by |x_E|:
  the zeroth order is the circular mean, the first order a linear ramp from the mean neighbour phase difference along
  each axis. Printed: |zeroth order|, the ramp across the 320-px FOV, and the circular std left after removing each.

usage: python reports/loss_domain/qc/qc_complex_maps.py [runs_root]  (default ~/data/seqmri_checkpoints/loss_domain/runs_v2)
"""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, 'scripts/eval'); sys.path.insert(0, '.')
import eval_loss_domain as e
from activemri.datasets.fastmri_data import build_manifest_dataloaders
from activemri.training.trainer import build_model
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img
from activemri.utils.metric_mask import metric_support_mask
from activemri.utils.metrics import masked_ssim

runs = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / 'data/seqmri_checkpoints/loss_domain/runs_v2'
device = torch.device('cuda')
_, val_loader = build_manifest_dataloaders(manifest='configs/datasets/fastmri_local_v1/manifest.csv', roots=e.ROOTS,
    classes=['FLAIR', 'knee'], image_shape=[320, 320], batch_size=16, num_coils=20, num_workers=8, complex_cache_dir=e.CACHE)
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_loader.dataset, list(range(0, len(val_loader.dataset), 12))),
                                     batch_size=16, num_workers=8, collate_fn=val_loader.collate_fn)
ck = torch.load(runs / e.run_name('rss', 42) / 'last_model.pt', map_location='cpu', weights_only=False)
model = build_model(ck['options']).to(device)
model.load_state_dict({k.removeprefix('module.'): v for k, v in ck['model'].items()}); model.eval()
model.sampler.eval_generator = None


def masked_psnr(output, target, mask):
    mse = ((output - target) ** 2 * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1))
    return 10 * torch.log10(target.flatten(1).amax(1) ** 2 / mse)


def circular(z, w):
    """(mean angle, circular std) of the unit phasors z weighted by w."""
    resultant = (w * z).sum() / w.sum()
    return torch.angle(resultant), torch.sqrt(-2 * torch.log(resultant.abs().clamp(min=1e-12)))


def ramp(z, w):
    """Mean phase step per pixel along rows and columns, from the weighted neighbour products of z."""
    gy = torch.angle((z[1:, :] * z[:-1, :].conj() * w[1:, :] * w[:-1, :]).sum())
    gx = torch.angle((z[:, 1:] * z[:, :-1].conj() * w[:, 1:] * w[:, :-1]).sum())
    return gy, gx


st = defaultdict(lambda: defaultdict(list))
torch.manual_seed(1234)
with torch.no_grad():
    for batch in loader:
        kr = batch['kspace'].to(device); t = batch['target'].to(device)
        k = torch.view_as_complex(kr.contiguous())
        xe = torch.view_as_complex(batch['complex_target'].to(device).contiguous())
        pred = model(t, kr)
        S, E = pred['sense_maps'], pred['kspace_extent']
        metric = metric_support_mask(k, list(batch['anatomy']))[:, 0]
        xl = (ifft_2d_img(k * E) * S.conj()).sum(dim=1)
        ceiling = ifft_2d_img(fft_2d_img(xl[:, None] * S) * E).abs().pow(2).sum(dim=1).sqrt().float()
        out = pred['output'][:, 0].float()
        rows = {
            'SSIM_Masked rss output': masked_ssim(out[:, None], t[:, None], metric[:, None]),
            'SSIM_Masked ceiling x_L': masked_ssim(ceiling[:, None], t[:, None], metric[:, None]),
            'PSNR_Masked rss output': masked_psnr(out, t, metric),
            'PSNR_Masked ceiling x_L': masked_psnr(ceiling, t, metric),
        }
        H, W = xl.shape[-2:]
        y = torch.arange(H, device=device, dtype=torch.float32)[:, None]
        x = torch.arange(W, device=device, dtype=torch.float32)[None, :]
        for i in range(len(t)):
            tissue = metric[i].bool() & (xe[i].abs() > 0.3 * xe[i].abs().max())
            w = xe[i].abs() * tissue
            z = torch.exp(1j * torch.angle(xl[i] * xe[i].conj()))
            mean, std0 = circular(z, w)
            gy, gx = ramp(z, w)
            flat = z * torch.exp(-1j * (gy * y + gx * x))
            _, std1 = circular(flat, w)
            for name, value in (('|zeroth order| (rad)', mean.abs()), ('circ std after zeroth order (rad)', std0),
                                ('ramp across FOV, y (rad)', gy.abs() * H), ('ramp across FOV, x (rad)', gx.abs() * W),
                                ('circ std after zeroth + first order (rad)', std1)):
                rows.setdefault(name, []).append(value)
        for name, values in rows.items():
            values = torch.stack(list(values)) if isinstance(values, list) else values
            for c, v in zip(batch['condition'], values.double().cpu().tolist()):
                st[c][name].append(v); st['all'][name].append(v)
for c in ('all', 'brain', 'knee-PD', 'knee-PDFS'):
    print(f'== {c} (n={len(st[c]["SSIM_Masked rss output"])})')
    for name, v in st[c].items():
        print(f'   {name:44s} mean {np.mean(v):8.4f}   median {np.median(v):8.4f}')
