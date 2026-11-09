"""Why the round-2 complex arm scores SSIM_Masked < 0.80: the rss and complex seed-42 checkpoints on matched train and val
subsets (every 24th train slice, every 12th val slice; eval mode, mask draw 1234).

Per arm and split: the scored metrics (RSS output vs RSS target, apodized metric mask), SSIM against the arm's own
target |x_E|, the complex loss as trained (sum_c conj(S_c) F^-1(F(S_c x) E) vs x_E, full-grid mean) with and without
the ESPIRiT phase handed to the recon, the phase error in tissue, data error on the sampled k-space, and the ceiling
x = x_E (scored through the arm's own maps and extent).

usage: python reports/loss_domain/diag/complex_ssim.py [runs_root]  (default ~/data/seqmri_checkpoints/loss_domain/runs_v2)
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
train_loader, val_loader = build_manifest_dataloaders(
    manifest='configs/datasets/fastmri_local_v1/manifest.csv', roots=e.ROOTS, classes=['FLAIR', 'knee'],
    image_shape=[320, 320], batch_size=16, num_coils=20, num_workers=8, complex_cache_dir=e.CACHE)
subsets = {split: torch.utils.data.DataLoader(
               torch.utils.data.Subset(loader.dataset, list(range(0, len(loader.dataset), stride))),
               batch_size=16, num_workers=8, collate_fn=val_loader.collate_fn)
           for split, loader, stride in (('train', train_loader, 24), ('val', val_loader, 12))}


def masked_psnr(output, target, mask):
    mse = ((output - target) ** 2 * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1))
    return 10 * torch.log10(target.flatten(1).amax(1) ** 2 / mse)


def scored(image, maps, extent):
    return ifft_2d_img(fft_2d_img(image[:, None] * maps) * extent).abs().pow(2).sum(dim=1).sqrt()


per_slice = {}
for arm in ('rss', 'complex'):
    ck = torch.load(runs / e.run_name(arm, 42) / 'last_model.pt', map_location='cpu', weights_only=False)
    model = build_model(ck['options']).to(device)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in ck['model'].items()}); model.eval()
    model.sampler.eval_generator = None
    for split, loader in subsets.items():
        torch.manual_seed(1234)
        st = defaultdict(lambda: defaultdict(list))
        with torch.no_grad():
            for batch in loader:
                kr = batch['kspace'].to(device); t = batch['target'].to(device)
                k = torch.view_as_complex(kr.contiguous())
                xe = torch.view_as_complex(batch['complex_target'].to(device).contiguous())
                pred = model(t, kr)
                S, x, E, mask = pred['sense_maps'], pred['recon_complex'][:, 0], pred['kspace_extent'], pred['mask']
                out = pred['output'][:, 0].float()
                metric = metric_support_mask(k, list(batch['anatomy']))[:, 0]
                combined = (ifft_2d_img(fft_2d_img(S * x[:, None]) * E) * S.conj()).sum(dim=1)
                data_combined = (ifft_2d_img(k) * S.conj()).sum(dim=1)
                tissue = metric.bool() & (xe.abs() > 0.3 * xe.abs().flatten(1).amax(1)[:, None, None])
                dphi = torch.angle(combined * xe.conj())
                given_phase = combined.abs() * torch.exp(1j * torch.angle(xe))
                ceiling = scored(xe, S, E)
                sampled = (mask * E).expand_as(k)
                data_err = ((fft_2d_img(S * x[:, None]) - k).abs().pow(2) * sampled).sum(dim=(1, 2, 3)) / \
                           (k.abs().pow(2) * sampled).sum(dim=(1, 2, 3))
                rows = {
                    'SSIM out vs RSS target (scored)': masked_ssim(out[:, None], t[:, None], metric[:, None]),
                    'PSNR out vs RSS target (scored)': masked_psnr(out, t, metric),
                    'SSIM out vs |x_E|': masked_ssim(out[:, None], xe.abs()[:, None].float(), metric[:, None]),
                    'SSIM ceiling x=x_E vs RSS target': masked_ssim(ceiling[:, None].float(), t[:, None], metric[:, None]),
                    'complex L1 as trained': (combined - xe).abs().flatten(1).mean(1),
                    'complex L1, ESPIRiT phase given': (given_phase - xe).abs().flatten(1).mean(1),
                    'median |dphi| recon vs x_E (tissue, rad)': torch.stack([dphi[i][tissue[i]].abs().median() for i in range(len(t))]),
                    'median |dphi| S^H coils vs x_E (tissue, rad)': torch.stack(
                        [torch.angle(data_combined[i] * xe[i].conj())[tissue[i]].abs().median() for i in range(len(t))]),
                    '|recon| / |x_E| (tissue, median)': torch.stack(
                        [(combined[i].abs() / xe[i].abs())[tissue[i]].median() for i in range(len(t))]),
                    'rel data err on sampled k': data_err.sqrt(),
                    'max |scored(x) - output| (FFT check)': (scored(x, S, E) - out).abs().flatten(1).amax(1),
                }
                for name, values in rows.items():
                    for c, v in zip(batch['condition'], values.double().cpu().tolist()):
                        st[c][name].append(v); st['all'][name].append(v)
        per_slice[arm, split] = st
        print(f'== {arm} / {split} (n={len(st["all"]["complex L1 as trained"])})')
        for name in st['all']:
            print(f'   {name:46s}' + '  '.join(f'{c} {np.mean(st[c][name]):7.4f}' for c in ('all', 'brain', 'knee-PD', 'knee-PDFS')))
    del model; torch.cuda.empty_cache()

for split in subsets:
    gap = np.array(per_slice['rss', split]['all']['SSIM out vs RSS target (scored)']) - \
          np.array(per_slice['complex', split]['all']['SSIM out vs RSS target (scored)'])
    phase = np.array(per_slice['complex', split]['all']['median |dphi| recon vs x_E (tissue, rad)'])
    rank = lambda a: np.argsort(np.argsort(a))
    print(f'{split}: Spearman(SSIM gap rss - complex, complex |dphi|) = {np.corrcoef(rank(gap), rank(phase))[0, 1]:.3f}')
