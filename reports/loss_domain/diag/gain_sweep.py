"""Which global gain g minimises each loss when applied to the RSS-L1 run's outputs (seed 42) on a val subset?"""
import sys
from collections import defaultdict
from pathlib import Path
import torch
sys.path.insert(0, 'scripts/eval'); sys.path.insert(0, '.')
import eval_loss_domain as e
from activemri.datasets.fastmri_data import build_manifest_dataloaders
from activemri.datasets.complex_image_cache import ComplexImageCache, cache_signature
from activemri.training.trainer import build_model
from activemri.utils.image_processing import fft_2d_img

device = torch.device('cuda')
_, val_loader = build_manifest_dataloaders(manifest='configs/datasets/fastmri_local_v1/manifest.csv', roots=e.ROOTS,
    classes=['FLAIR', 'knee'], image_shape=[320, 320], batch_size=16, num_coils=20, num_workers=8)
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_loader.dataset, list(range(0, len(val_loader.dataset), 24))),
                                     batch_size=16, num_workers=8, collate_fn=val_loader.collate_fn)
cache = ComplexImageCache(e.CACHE)
ck = torch.load(Path.home() / 'data/seqmri_checkpoints/loss_domain/runs' / e.run_name('rss', 42) / 'last_model.pt',
                map_location='cpu', weights_only=False)
model = build_model(ck['options']).to(device)
model.load_state_dict({k.removeprefix('module.'): v for k, v in ck['model'].items()}); model.eval()
gains = torch.linspace(0.5, 1.2, 36, device=device)
totals = defaultdict(lambda: defaultdict(lambda: torch.zeros_like(gains)))
torch.manual_seed(1234)
with torch.no_grad():
    for batch in loader:
        kr = batch['kspace'].to(device); t = batch['target'].to(device)
        o = model(t, kr)['output'][:, 0].float()
        M = torch.stack([cache.load(a, f, int(s), cache_signature(a, [320, 320], 20, 0.2)).abs() > 0
                         for a, f, s in zip(batch['anatomy'], batch['fname'], batch['slice_id'])]).to(device).float()
        for i, g in enumerate(gains):
            losses = {'image L1': (g * o - t).abs().mean(dim=(-2, -1)),
                      'kspace L1': (fft_2d_img(g * o) - fft_2d_img(t)).abs().mean(dim=(-2, -1)),
                      'kspace L1 in M': (fft_2d_img(g * o * M) - fft_2d_img(t * M)).abs().mean(dim=(-2, -1)),
                      'image L2': ((g * o - t) ** 2).mean(dim=(-2, -1))}
            for name, per_slice in losses.items():
                for c, v in zip(batch['condition'], per_slice):
                    totals[name][c][i] += v; totals[name]['all'][i] += v
for name, by_c in totals.items():
    print(f"{name:15s} " + '  '.join(f"{c} g*={gains[v.argmin()].item():.2f}" for c, v in sorted(by_c.items())))
