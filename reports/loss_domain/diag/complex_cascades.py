"""Where in the 12 VarNet cascades the round-2 complex arm takes the ESPIRiT phase, and what that costs: the coil
k-space after each cascade of the rss and complex seed-42 checkpoints (every 12th val slice, eval mode, mask draw 1234)
is coil-combined with the arm's maps, x_i = sum_c conj(S_c) F^-1(k_i,c), and scored like the final output.

Per cascade: SSIM_Masked of the scored RSS image against the RSS target, median |angle(x_i conj(x_E))| in tissue, and
the relative error of k_i on the sampled k-space. Each cascade's learned DC weight is printed with it.

usage: python reports/loss_domain/diag/complex_cascades.py [runs_root]  (default ~/data/seqmri_checkpoints/loss_domain/runs_v2)
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

for arm in ('rss', 'complex'):
    ck = torch.load(runs / e.run_name(arm, 42) / 'last_model.pt', map_location='cpu', weights_only=False)
    model = build_model(ck['options']).to(device)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in ck['model'].items()}); model.eval()
    model.sampler.eval_generator = None
    cascades = model.reconstructor.cascades
    captured = []
    for cascade in cascades:
        cascade.register_forward_hook(lambda module, inputs, output: captured.append(output))
    torch.manual_seed(1234)
    st = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for batch in loader:
            captured.clear()
            kr = batch['kspace'].to(device); t = batch['target'].to(device)
            k = torch.view_as_complex(kr.contiguous())
            xe = torch.view_as_complex(batch['complex_target'].to(device).contiguous())
            pred = model(t, kr)
            S, E, mask = pred['sense_maps'], pred['kspace_extent'], pred['mask']
            metric = metric_support_mask(k, list(batch['anatomy']))[:, 0]
            tissue = metric.bool() & (xe.abs() > 0.3 * xe.abs().flatten(1).amax(1)[:, None, None])
            sampled = (mask * E).expand_as(k)
            for i, coil_kspace in enumerate(captured):
                x = (ifft_2d_img(coil_kspace) * S.conj()).sum(dim=1)
                image = ifft_2d_img(fft_2d_img(x[:, None] * S) * E).abs().pow(2).sum(dim=1).sqrt()
                dphi = torch.angle(x * xe.conj())
                rows = {
                    'SSIM': masked_ssim(image[:, None].float(), t[:, None], metric[:, None]),
                    '|dphi| vs x_E': torch.stack([dphi[j][tissue[j]].abs().median() for j in range(len(t))]),
                    'data err': (((coil_kspace - k).abs().pow(2) * sampled).sum(dim=(1, 2, 3)) /
                                 (k.abs().pow(2) * sampled).sum(dim=(1, 2, 3))).sqrt(),
                }
                for name, values in rows.items():
                    for c, v in zip(batch['condition'], values.double().cpu().tolist()):
                        st[c, name][i].append(v); st['all', name][i].append(v)
            final = (ifft_2d_img(captured[-1]) * S.conj()).sum(dim=1)
            assert torch.allclose(final, pred['recon_complex'][:, 0], atol=1e-4), 'last hook != recon_complex'
    weights = [float(c.dc_weight.flatten()[0]) for c in cascades]
    print(f'== {arm}: cascade  dc_weight  ' + '  '.join(f'{c} SSIM / |dphi| / data err' for c in ('all', 'brain', 'knee-PD', 'knee-PDFS')))
    for i, w in enumerate(weights):
        print(f'   {i:2d}  {w:7.3f}  ' + '  '.join(
            f"{np.mean(st[c, 'SSIM'][i]):.3f} / {np.mean(st[c, '|dphi| vs x_E'][i]):.2f} / {np.mean(st[c, 'data err'][i]):.2f}"
            for c in ('all', 'brain', 'knee-PD', 'knee-PDFS')))
    del model; torch.cuda.empty_cache()
