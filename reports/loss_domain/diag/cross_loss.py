"""(1) Cross-evaluate every training loss on every seed-42 model (eval mode, val subset): does each arm reach the lowest
value of its own loss? (2) ESPIRiT scale vs the RSS target. (3) k-space extent coverage and where kspace_rss's error is."""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, 'scripts/eval'); sys.path.insert(0, '.')
import eval_loss_domain as e
from activemri.datasets.fastmri_data import build_manifest_dataloaders
from activemri.models.sequential_sampling import SequentialSampling
from activemri.training.trainer import build_model
from activemri.utils.image_processing import fft_2d_img
from activemri.utils.metric_mask import metric_support_mask

out_dir = Path(sys.argv[1])
device = torch.device('cuda')
_, val_loader = build_manifest_dataloaders(manifest='configs/datasets/fastmri_local_v1/manifest.csv', roots=e.ROOTS,
    classes=['FLAIR', 'knee'], image_shape=[320, 320], batch_size=16, num_coils=20, num_workers=8,
    complex_cache_dir=e.CACHE)
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_loader.dataset, list(range(0, len(val_loader.dataset), 12))),
                                     batch_size=16, num_workers=8, collate_fn=val_loader.collate_fn)
LOSSES = {'image L1 (rss)': None, 'complex_l1': ('complex_l1', False), 'kspace_sense_l1': ('kspace_sense_l1', False),
          'kspace_rss_l1': ('kspace_rss_l1', False), 'kspace_rss_l1 + M': ('kspace_rss_l1', True)}
cross = {}
scale = defaultdict(lambda: defaultdict(list))
saved = {}
for arm in ('rss', 'complex', 'kspace_rss', 'kspace_rss_M'):
    ck = torch.load(Path.home() / 'data/seqmri_checkpoints/loss_domain/runs' / e.run_name(arm, 42) / 'last_model.pt',
                    map_location='cpu', weights_only=False)
    model = build_model(ck['options']).to(device)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in ck['model'].items()}); model.eval()
    model.sampler.eval_generator = None
    torch.manual_seed(1234)
    sums = defaultdict(float); n = 0
    with torch.no_grad():
        for b, batch in enumerate(loader):
            kr = batch['kspace'].to(device); t = batch['target'].to(device)[:, None]
            xe = torch.view_as_complex(batch['complex_target'].to(device).contiguous())[:, None]
            pred = model(t[:, 0], kr)
            for name, spec in LOSSES.items():
                if spec is None:
                    v = (pred['output'] - t).abs().mean()
                else:
                    v = SequentialSampling.loss_domain_l1(pred, t, xe, *spec)
                sums[name] += float(v) * len(t)
            n += len(t)
            if arm == 'rss':
                m = metric_support_mask(torch.view_as_complex(kr.contiguous()), list(batch['anatomy']))[:, :1].bool()
                inside = m & (xe.abs() > 0)
                E = pred['kspace_extent']
                T = fft_2d_img(t)
                for i, c in enumerate(batch['condition']):
                    r = (xe.abs()[i][inside[i]] / t[i][inside[i]])
                    scale[c]['ratio p10/p50/p90'].append(torch.quantile(r, torch.tensor([.1, .5, .9], device=device)).cpu().numpy())
                    scale[c]['norm ratio |xE|/t in mask'].append(float((xe.abs()[i] * m[i]).norm() / (t[i] * m[i]).norm()))
                    scale[c]['t max'].append(float(t[i].max())); scale[c]['|xE| max'].append(float(xe[i].abs().max()))
                    scale[c]['M frac of grid'].append(float((xe[i].abs() > 0).float().mean()))
                    scale[c]['metric mask in M'].append(float((m[i] & (xe[i].abs() > 0)).sum() / m[i].sum()))
                    scale[c]['E frac of grid'].append(float(E[i].mean()))
                    scale[c]['target k-energy outside E'].append(float((T[i].abs() ** 2 * (1 - E[i])).sum() / (T[i].abs() ** 2).sum()))
            if b == 4 and arm in ('rss', 'kspace_rss', 'complex'):
                saved[arm] = {'out': pred['output'][:, 0].float().cpu(), 't': t[:, 0].cpu(), 'xe': xe[:, 0].cpu(),
                              'cond': list(batch['condition']), 'E': pred['kspace_extent'][:, 0].cpu()}
    cross[arm] = {k: v / n for k, v in sums.items()}
    del model; torch.cuda.empty_cache()

print('Cross-evaluated losses (rows: model, cols: loss; * = lowest in column)')
cols = list(LOSSES)
print(f"{'model':14s}" + ''.join(f'{c:>20s}' for c in cols))
best = {c: min(cross, key=lambda a: cross[a][c]) for c in cols}
for arm, row in cross.items():
    print(f'{arm:14s}' + ''.join(f"{row[c]:>19.5f}{'*' if best[c] == arm else ' '}" for c in cols))
print('\nESPIRiT scale and extent (means over the subset)')
for c, d in sorted(scale.items()):
    print(c)
    for k, v in d.items():
        print(f'   {k:28s} {np.round(np.mean(np.stack(v), axis=0), 3)}')
torch.save(saved, out_dir / 'cross_loss_examples.pt')
