"""k-space / coil-image consistency of the ESPIRiT and learned-map representations, and data consistency of the
seed-42 rss and complex outputs (val subset; ESPIRiT maps recomputed with BART)."""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, 'scripts/eval'); sys.path.insert(0, '.')
import eval_loss_domain as e
from activemri.datasets.fastmri_data import build_manifest_dataloaders
from activemri.training.trainer import build_model
from activemri.utils.bart_ecalib import PRESETS, ecalib_sense_maps
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img
from activemri.utils.metric_mask import metric_support_mask

device = torch.device('cuda')
_, val_loader = build_manifest_dataloaders(manifest='configs/datasets/fastmri_local_v1/manifest.csv', roots=e.ROOTS,
    classes=['FLAIR', 'knee'], image_shape=[320, 320], batch_size=16, num_coils=20, num_workers=8, complex_cache_dir=e.CACHE)
n = len(val_loader.dataset)
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_loader.dataset, list(range(7, n, n // 24))),
                                     batch_size=8, num_workers=8, collate_fn=val_loader.collate_fn)
models = {}
for arm in ('rss', 'complex'):
    ck = torch.load(Path.home() / 'data/seqmri_checkpoints/loss_domain/runs' / e.run_name(arm, 42) / 'last_model.pt',
                    map_location='cpu', weights_only=False)
    m = build_model(ck['options']).to(device)
    m.load_state_dict({k.removeprefix('module.'): v for k, v in ck['model'].items()}); m.eval(); m.sampler.eval_generator = None
    models[arm] = m


def rel(a, b, w):
    return float(((a - b).abs() ** 2 * w).sum() / ((b.abs() ** 2) * w).sum()) ** 0.5


def phase_gradient(z, w):
    gy = torch.angle((z[1:, :] * z[:-1, :].conj() * (w[1:, :] * w[:-1, :])).sum())
    gx = torch.angle((z[:, 1:] * z[:, :-1].conj() * (w[:, 1:] * w[:, :-1])).sum())
    return float(gy), float(gx)


st = defaultdict(lambda: defaultdict(list))
torch.manual_seed(1234)
with torch.no_grad():
    for batch in loader:
        kr = batch['kspace'].to(device); t = batch['target'].to(device)
        k = torch.view_as_complex(kr.contiguous())
        coils = ifft_2d_img(k)
        xe_cache = torch.view_as_complex(batch['complex_target'].to(device).contiguous())
        metric = metric_support_mask(k, list(batch['anatomy']))[:, 0].bool()
        preds = {arm: m(t, kr) for arm, m in models.items()}
        for i, (c, anatomy) in enumerate(zip(batch['condition'], batch['anatomy'])):
            SE, _ = ecalib_sense_maps(k[i].cpu(), **PRESETS[anatomy]); SE = SE.to(device)
            xe = (coils[i] * SE.conj()).sum(0)
            SL = preds['rss']['sense_maps'][i]; xl = (coils[i] * SL.conj()).sum(0)
            inside = (SE.abs().pow(2).sum(0) > 0.5) & (SL.abs().pow(2).sum(0) > 0.5)
            tissue = inside & metric[i] & (xe.abs() > 0.3 * xe.abs().max())
            w = tissue.float()
            d = st[c]
            d['cache x_E vs recomputed'].append(rel(xe_cache[i], xe, inside.float()))
            d['coil err S_E·x_E (own maps)'].append(rel(SE * xe, coils[i], w))
            d['coil err S_L·x_L (own maps)'].append(rel(SL * xl, coils[i], w))
            d['coil err S_L·x_E (kspace_sense target)'].append(rel(SL * xe, coils[i], w))
            coil_phase = torch.angle(SL * SE.conj())                                 # [C, H, W]
            weight = (SL.abs() * SE.abs()) * w
            resultant = (weight * torch.exp(1j * coil_phase)).sum(0).abs() / weight.sum(0).clamp(min=1e-12)
            d['coil spread of ∠(S_L·conj S_E) (rad)'].append(float((torch.sqrt(-2 * torch.log(resultant.clamp(1e-6, 1))) * w).sum() / w.sum()))
            gy, gx = phase_gradient(xl * xe.conj(), w)
            d['phase ramp of x_L·conj x_E (×2π/320 per px) y'].append(gy / (2 * np.pi / 320))
            d['phase ramp of x_L·conj x_E (×2π/320 per px) x'].append(gx / (2 * np.pi / 320))
            for arm, p in preds.items():
                S, x, E, mask = p['sense_maps'][i], p['recon_complex'][i, 0], p['kspace_extent'][i, 0], p['mask'][i, 0]
                pk = fft_2d_img(S * x) * E
                d[f'{arm}: data err on sampled k'].append(rel(pk, k[i], mask.expand_as(pk)))
                d[f'{arm}: err vs full k, unsampled'].append(rel(pk, k[i], ((1 - mask) * E).expand_as(pk)))
for c, d in sorted(st.items()):
    print(f'== {c} (n={len(next(iter(d.values())))})')
    for key, v in d.items():
        print(f'   {key:48s} {np.mean(v):8.4f}')
