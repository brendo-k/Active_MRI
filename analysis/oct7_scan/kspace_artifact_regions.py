"""Where in k-space does a reconstruction differ from the CG-SENSE solution of all acquired data?

Regions: lattice (acquired points Omega), hull-off-lattice (inside the acquired envelope but never measured),
outside the hull (partial-Fourier band, elliptical corners). SSDU only ever scores the first.
"""
import sys
sys.path.insert(0, '.')
import numpy as np, torch
import torch.nn.functional as F
from activemri.datasets.siemens_space_data import SiemensSpaceDataset
from activemri.reconstructors.CG_Sense import conjugate_gradient_sense
from activemri.training.trainer import build_model
from activemri.utils.image_processing import fft_2d_img

checkpoint_path = sys.argv[1]
device = 'cuda'
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
model = build_model(ckpt['options']).to(device)
model.load_state_dict({k.removeprefix('module.'): v for k, v in ckpt['model'].items()}); model.eval()
path = '/home/bkadota/data/my_scans/oct_7_2026_cache/meas_MID00295_FID30725_bk_space.h5'
ds = SiemensSpaceDataset([path], 'val')
omega = ds.omegas[ds.examples[0][0]].to(device)
hull = (F.max_pool2d(omega[None, None], 3, 1, 1)[0, 0] > 0).float()
regions = {'lattice (Omega)': omega, 'hull, off-lattice': hull * (1 - omega), 'outside hull': 1 - hull}
print({k: int(v.sum()) for k, v in regions.items()})
batch = torch.utils.data.default_collate([ds[i] for i in (3, 10, 17)])
with torch.no_grad():
    out = model(batch['target'].to(device), batch['kspace'].to(device), sense_maps=batch['sense_maps'].to(device))
    S = batch['sense_maps'].to(device)
    full = torch.view_as_complex(batch['kspace'].to(device)) + torch.view_as_complex(batch['holdout_kspace'].to(device))
    ref = conjugate_gradient_sense(full, S, omega[None, None].expand(3, 1, -1, -1), 30, 0.01)
    mine = out['recon_complex'][:, 0]
    for name, img in (('VarNet - CG(all)', mine - ref), ('CG(all) itself', ref), ('VarNet itself', mine)):
        e = fft_2d_img(img).abs().pow(2)
        total = e.sum((-2, -1))
        print(name.ljust(18), '  '.join(f'{r}: {((e * m).sum((-2, -1)) / total).mean():.3f}' for r, m in regions.items()))
    ky = fft_2d_img(mine - ref).abs().pow(2).sum((0, -1)) 
    top = torch.topk(ky, 6).indices.sort().values - 128
    print('VarNet-CG(all): top ky bins (relative to centre):', top.tolist(), '| Omega rows (min,max):', (omega.sum(1).nonzero().min().item() - 128, omega.sum(1).nonzero().max().item() - 128))
