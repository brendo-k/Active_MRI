"""Complex-L1 arm (seed 42) vs the RSS arm on a val subset: masks in the complex loss, phase error against ESPIRiT,
magnitude shrinkage, and the loss with the phase error removed."""
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

device = torch.device('cuda')
_, val_loader = build_manifest_dataloaders(manifest='configs/datasets/fastmri_local_v1/manifest.csv', roots=e.ROOTS,
    classes=['FLAIR', 'knee'], image_shape=[320, 320], batch_size=16, num_coils=20, num_workers=8, complex_cache_dir=e.CACHE)
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_loader.dataset, list(range(0, len(val_loader.dataset), 12))),
                                     batch_size=16, num_workers=8, collate_fn=val_loader.collate_fn)
for arm in ('complex', 'rss'):
    ck = torch.load(Path.home() / 'data/seqmri_checkpoints/loss_domain/runs' / e.run_name(arm, 42) / 'last_model.pt',
                    map_location='cpu', weights_only=False)
    model = build_model(ck['options']).to(device)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in ck['model'].items()}); model.eval()
    model.sampler.eval_generator = None
    torch.manual_seed(1234)
    st = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for batch in loader:
            kr = batch['kspace'].to(device); t = batch['target'].to(device)
            k = torch.view_as_complex(kr.contiguous())
            xe = torch.view_as_complex(batch['complex_target'].to(device).contiguous())
            pred = model(t, kr)
            S, x, E = pred['sense_maps'], pred['recon_complex'][:, 0], pred['kspace_extent']
            combined = (ifft_2d_img(fft_2d_img(S * x[:, None]) * E) * S.conj()).sum(dim=1)
            omega = S.abs().pow(2).sum(dim=1) > 0.5
            M = xe.abs() > 0
            metric = metric_support_mask(k, list(batch['anatomy']))[:, 0].bool()
            tissue = metric & omega & (xe.abs() > 0.3 * xe.abs().flatten(1).amax(1)[:, None, None])
            full_comb = (ifft_2d_img(k) * S.conj()).sum(dim=1)   # fully sampled data, learned maps
            err = (combined - xe).abs()
            dphi = torch.angle(combined * xe.conj())
            rotated = combined.abs() * torch.exp(1j * torch.angle(xe))
            for i, c in enumerate(batch['condition']):
                for key in (c, 'all'):
                    d = st[key]
                    d['loss share outside Ω (constant)'].append(float(err[i][~omega[i]].sum() / err[i].sum()))
                    d['metric mask inside Ω'].append(float((metric[i] & omega[i]).sum() / metric[i].sum()))
                    d['Ω share of grid'].append(float(omega[i].float().mean())); d['M share of grid'].append(float(M[i].float().mean()))
                    d['median |Δφ| recon vs ESPIRiT (tissue)'].append(float(dphi[i][tissue[i]].abs().median()))
                    d['median |angle| of Σconj(S)·coils (tissue)'].append(float(torch.angle(full_comb[i][tissue[i]]).abs().median()))
                    d['median |angle| of x_E (tissue)'].append(float(torch.angle(xe[i][tissue[i]]).abs().median()))
                    d['|recon| / |x_E| (tissue)'].append(float((combined[i].abs()[tissue[i]] / xe[i].abs()[tissue[i]]).median()))
                    d['cos Δφ (tissue)'].append(float(torch.cos(dphi[i][tissue[i]]).median()))
                    d['complex L1 as is'].append(float(err[i].mean()))
                    d['complex L1, recon given ESPIRiT phase'].append(float((rotated[i] - xe[i]).abs().mean()))
    print(f'== {arm}')
    for c in ('all', 'brain', 'knee-PD', 'knee-PDFS'):
        print('  ' + c + ': ' + '  '.join(f"{k} {np.mean(v):.3f}" for k, v in st[c].items()))
    del model; torch.cuda.empty_cache()
