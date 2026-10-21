"""Do the dataset's k-space, sense maps and PICS reference agree under the repo's own FFT/SENSE code?

Classical CG-SENSE from (a) the input pool Omega\\Lambda and (b) all of Omega, scored against the PICS reference
the way the trainer scores (signal region), plus the SSDU loss of (a) on Lambda. No training involved.
"""
import sys
sys.path.insert(0, '.')
import torch
from activemri.datasets.siemens_space_data import SiemensSpaceDataset
from activemri.reconstructors.CG_Sense import conjugate_gradient_sense
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.evaluate import build_metrics
from activemri.utils.image_processing import fft_2d_img

device = 'cuda'
dataset = SiemensSpaceDataset(['/home/bkadota/data/my_scans/oct_7_2026_cache/meas_MID00295_FID30725_bk_space.h5'], 'val')
loader = torch.utils.data.DataLoader(dataset, batch_size=8)
rows = {}
for iters, lamda in ((10, 0.0), (30, 0.0), (30, 1e-3), (30, 1e-2), (30, 1e-1)):
    metrics = {name: build_metrics(device) for name in ('pool', 'all')}
    ssdu = []
    for batch in loader:
        sense = batch['sense_maps'].to(device)
        pool_k = torch.view_as_complex(batch['kspace'].to(device))
        hold_k = torch.view_as_complex(batch['holdout_kspace'].to(device))
        hold_mask = batch['holdout_mask'].to(device)
        pool = (pool_k.abs().sum(1, keepdim=True) > 0).float()
        target = batch['target'].to(device)
        for name, k, m in (('pool', pool_k, pool), ('all', pool_k + hold_k, pool + hold_mask)):
            x = conjugate_gradient_sense(k, sense, m, num_iters=iters, lamda=lamda)
            out = (x.abs() * (sense.abs().pow(2).sum(1) > 0).float()).unsqueeze(1)
            metrics[name]['PSNR'].data_range = metrics[name]['SSIM'].data_range = target.max()
            mo, mt = _apply_signal_mask(out, target.unsqueeze(1))
            metrics[name].update(mo, mt)
            if name == 'pool':
                err = ((fft_2d_img(x.unsqueeze(1) * sense) - hold_k) * hold_mask).abs().flatten(1)
                ref = hold_k.abs().flatten(1)
                ssdu.append(err.pow(2).sum(1).sqrt() / ref.pow(2).sum(1).sqrt() + err.sum(1) / ref.sum(1))
    print(f'CG-SENSE iters={iters} lamda={lamda:g}  SSDU(pool->Lambda)={torch.cat(ssdu).mean():.4f}')
    for name, m in metrics.items():
        print('   ', name.ljust(5), ' '.join(f'{k}={v.item():.4f}' for k, v in m.compute().items()))
