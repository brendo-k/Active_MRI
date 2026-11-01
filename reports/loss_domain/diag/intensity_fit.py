"""Per-slice intensity diagnostics of the seed-42 loss-domain runs on a val subset: raw PSNR_Masked, after a
least-squares gain, after gain+offset, the output/target mean ratio, and the share of masked error energy at low
k-space frequency (radius < 16 px)."""
import sys
from collections import defaultdict
from pathlib import Path
import torch
sys.path.insert(0, 'scripts/eval'); sys.path.insert(0, '.')
import eval_loss_domain as e
from activemri.datasets.fastmri_data import build_manifest_dataloaders
from activemri.training.trainer import build_model
from activemri.utils.metric_mask import metric_support_mask

device = torch.device('cuda')
runs = Path.home() / 'data/seqmri_checkpoints/loss_domain/runs'
_, val_loader = build_manifest_dataloaders(manifest='configs/datasets/fastmri_local_v1/manifest.csv', roots=e.ROOTS,
    classes=['FLAIR', 'knee'], image_shape=[320, 320], batch_size=16, num_coils=20, num_workers=8)
subset = torch.utils.data.Subset(val_loader.dataset, list(range(0, len(val_loader.dataset), 12)))
loader = torch.utils.data.DataLoader(subset, batch_size=16, num_workers=8, collate_fn=val_loader.collate_fn)
yy, xx = torch.meshgrid(torch.arange(320) - 160, torch.arange(320) - 160, indexing='ij')
low = ((yy ** 2 + xx ** 2) < 16 ** 2).to(device)

def psnr(o, t, m):
    return 10 * torch.log10(t.amax(dim=(-2, -1)) ** 2 / (((t - o) ** 2 * m).sum(dim=(-2, -1)) / m.sum(dim=(-2, -1))))

for arm in ('rss', 'complex', 'kspace_rss', 'kspace_rss_M'):
    ck = torch.load(runs / e.run_name(arm, 42) / 'last_model.pt', map_location='cpu', weights_only=False)
    model = build_model(ck['options']).to(device)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in ck['model'].items()})
    model.eval(); model.sampler.eval_generator = None
    torch.manual_seed(1234)
    stats = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for batch in loader:
            kr = batch['kspace'].to(device); t = batch['target'].to(device)
            m = metric_support_mask(torch.view_as_complex(kr.contiguous()), list(batch['anatomy']))[:, 0]
            o = model(t, kr)['output'][:, 0].float()
            s = lambda x: (x * m).sum(dim=(-2, -1))
            gain = s(o * t) / s(o * o)
            n = s(torch.ones_like(o)); mo, mt = s(o) / n, s(t) / n
            cov = s((o - mo[:, None, None]) * (t - mt[:, None, None])); var = s((o - mo[:, None, None]) ** 2)
            a = cov / var; b = mt - a * mo
            err = torch.fft.fftshift(torch.fft.fft2((o - t) * m, norm='ortho'), dim=(-2, -1)).abs() ** 2
            for c, *v in zip(batch['condition'], psnr(o, t, m), psnr(gain[:, None, None] * o, t, m),
                             psnr(a[:, None, None] * o + b[:, None, None], t, m), mo / mt,
                             (err * low).sum(dim=(-2, -1)) / err.sum(dim=(-2, -1))):
                for k, x in zip(('raw', 'gain', 'gain+off', 'mean ratio', 'lowfreq err'), v):
                    stats[c][k].append(float(x)); stats['all'][k].append(float(x))
    for c in ('all', 'brain', 'knee-PD', 'knee-PDFS'):
        print(f"{arm:13s} {c:9s} n={len(stats[c]['raw']):3d} " + '  '.join(
            f"{k} {sum(v) / len(v):.3f}" for k, v in stats[c].items()), flush=True)
