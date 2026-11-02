"""Sizes and overlaps of the three image-space masks, per condition, on a val subset (fixed loader, v2 cache).

- metric: the trainer's apodized metric mask (activemri/utils/metric_mask.py).
- M: the ESPIRiT support, |x_espirit| > 0 (BART eigenvalue crop).
- Omega: the sensitivity net's pre_network mask, RSS of the 10x10 k-space centre > 0.1 max (maps are zero outside).
Also the RSS target's noise floor (median of t) in each region outside the smaller masks.
"""
import sys
from collections import defaultdict

import torch

sys.path.insert(0, 'scripts/eval'); sys.path.insert(0, '.')
import eval_loss_domain as e  # noqa: E402
from activemri.datasets.fastmri_data import build_manifest_dataloaders  # noqa: E402
from activemri.utils.image_processing import ifft_2d_img  # noqa: E402
from activemri.utils.metric_mask import metric_support_mask  # noqa: E402

_, val_loader = build_manifest_dataloaders(manifest='configs/datasets/fastmri_local_v1/manifest.csv', roots=e.ROOTS,
    classes=['FLAIR', 'knee'], image_shape=[320, 320], batch_size=16, num_coils=20, num_workers=8, complex_cache_dir=e.CACHE)
n = len(val_loader.dataset)
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_loader.dataset, list(range(3, n, n // 120))),
                                     batch_size=8, num_workers=16, collate_fn=val_loader.collate_fn)


def omega(k):
    centre = torch.zeros_like(k)
    h, w = k.shape[-2] // 2, k.shape[-1] // 2
    centre[..., h - 5:h + 5, w - 5:w + 5] = 1
    rss = ifft_2d_img(k * centre).abs().pow(2).sum(1).sqrt()
    return rss > 0.1 * rss.amax(dim=(-2, -1), keepdim=True)


stats = defaultdict(lambda: defaultdict(list))
with torch.no_grad():
    for batch in loader:
        k = torch.view_as_complex(batch['kspace'].contiguous())
        t = batch['target']
        masks = {'metric': metric_support_mask(k, list(batch['anatomy']))[:, 0].bool(),
                 'M': torch.view_as_complex(batch['complex_target'].contiguous()).abs() > 0,
                 'Omega': omega(k)}
        for i, condition in enumerate(batch['condition']):
            s = stats[condition]
            m = {name: mask[i] for name, mask in masks.items()}
            for name, mask in m.items():
                s[f'{name}: share of grid'].append(mask.float().mean().item())
            for a in m:
                for b in m:
                    if a != b:
                        s[f'{a} inside {b}'].append((m[a] & m[b]).sum().item() / max(m[a].sum().item(), 1))
            s['target floor outside M (median t)'].append(t[i][~m['M']].median().item() if (~m['M']).any() else float('nan'))
            s['target floor in M minus Omega (median t)'].append(
                t[i][m['M'] & ~m['Omega']].median().item() if (m['M'] & ~m['Omega']).any() else float('nan'))
            s['target in metric mask (median t)'].append(t[i][m['metric']].median().item())

for condition, s in sorted(stats.items()):
    print(f'== {condition} (n={len(s["M: share of grid"])})')
    for key, values in s.items():
        values = torch.tensor(values)
        values = values[~values.isnan()]
        print(f'   {key:42s} {values.mean().item():.3f}  (min {values.min().item():.3f})')

# Check against the model: Omega from the seed-42 rss run's actual maps, and what the scored output does in
# metric \ Omega (round-1 checkpoint, trained on the old brain flip, so brain is approximate).
if torch.cuda.is_available():
    from pathlib import Path
    from activemri.training.trainer import build_model
    device = torch.device('cuda')
    ck = torch.load(Path.home() / 'data/seqmri_checkpoints/loss_domain/runs' / e.run_name('rss', 42) / 'last_model.pt',
                    map_location='cpu', weights_only=False)
    model = build_model(ck['options']).to(device)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in ck['model'].items()}); model.eval()
    model.sampler.eval_generator = None
    check = defaultdict(lambda: defaultdict(list))
    torch.manual_seed(1234)
    with torch.no_grad():
        for batch in loader:
            kr, t = batch['kspace'].to(device), batch['target'].to(device)
            k = torch.view_as_complex(kr.contiguous())
            pred = model(t, kr)
            o = pred['output'][:, 0] if pred['output'].dim() == 4 else pred['output']
            model_omega = pred['sense_maps'].abs().pow(2).sum(1) > 0.5
            metric = metric_support_mask(k, list(batch['anatomy']))[:, 0].bool()
            ours = omega(k)
            for i, condition in enumerate(batch['condition']):
                c = check[condition]
                c['model Omega == reimplemented Omega'].append((model_omega[i] == ours[i]).float().mean().item())
                c['metric inside model Omega'].append((metric[i] & model_omega[i]).sum().item() / metric[i].sum().item())
                outside = metric[i] & ~model_omega[i]
                err = (o[i] - t[i]).abs()
                c['share of metric-mask L1 error from metric \\ Omega'].append(err[outside].sum().item() / err[metric[i]].sum().item())
                if outside.any():
                    c['output / target in metric \\ Omega (median)'].append((o[i][outside] / t[i][outside].clamp(min=1e-6)).median().item())
                    c['target in metric \\ Omega (median t)'].append(t[i][outside].median().item())
    for condition, c in sorted(check.items()):
        print(f'== model check, {condition}')
        for key, values in c.items():
            values = torch.tensor(values)
            print(f'   {key:50s} {values.mean().item():.3f}  (min {values.min().item():.3f})')
