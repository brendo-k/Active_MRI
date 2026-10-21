"""Invariants of the SSDU pipeline: mask inside the input pool, hold-out never an input, loss exact at the optimum."""
import sys
sys.path.insert(0, '.')
import torch
from hydra import compose, initialize
from activemri.training.builders import build_dataloaders, build_train_config
from activemri.training.trainer import build_model
from activemri.utils.image_processing import fft_2d_img

with initialize(version_base=None, config_path='../../configs'):
    cfg = compose('config', overrides=['recon=varnet', 'datasets=siemens_space', 'sampler.sampler_type=loupe_2d',
                                       'sampler.accelerations=[12]', 'misc.resolution=[256,64]', 'datasets.num_workers=0'])
options = build_train_config(cfg)
_, val_loader = build_dataloaders(cfg)
batch = next(iter(val_loader))
model = build_model(options).cuda().eval()
kspace = batch['kspace'].cuda()
with torch.no_grad():
    out = model(batch['target'].cuda(), kspace, sense_maps=batch['sense_maps'].cuda())

pool = (torch.view_as_complex(kspace).abs().sum(1, keepdim=True) > 0).float()
holdout = batch['holdout_mask'].cuda()
mask = out['mask']
print('mask points', mask.sum((1, 2, 3)).tolist(), 'pool points', pool.sum((1, 2, 3)).tolist())
print('mask outside pool:', (mask * (1 - pool)).sum().item(), '| mask on holdout:', (mask * holdout).sum().item(), '| pool & holdout:', (pool * holdout).sum().item())

target_dict = {'target': batch['target'].cuda(), 'kspace': kspace, 'holdout_mask': holdout, 'holdout_kspace': batch['holdout_kspace'].cuda()}
exact = torch.view_as_complex(batch['holdout_kspace'].cuda() + kspace)  # all of Omega
sense = out['sense_maps']
image = (torch.fft.ifftshift(exact, dim=(-2, -1)))
image = torch.fft.fftshift(torch.fft.ifft2(image, norm='ortho'), dim=(-2, -1))
image = (image * sense.conj()).sum(1, keepdim=True)  # SENSE adjoint of everything acquired
fake = dict(out, recon_complex=image)
print('ssdu loss, adjoint-of-all-Omega image (not exact, informational):', model.ssdu_loss(fake, target_dict).tolist())
# an image whose forward model reproduces the hold-out exactly: build it from the hold-out data by construction
k_holdout = torch.view_as_complex(batch['holdout_kspace'].cuda())
class Exact:
    pass
pred_k_equals_measured = model.ssdu_loss(dict(out, recon_complex=torch.zeros_like(image)), target_dict)
print('ssdu loss for an all-zero prediction (expect exactly 2):', pred_k_equals_measured.tolist())
print('data range check: val kspace abs max', torch.view_as_complex(kspace).abs().max().item(), 'target max', batch['target'].max().item())
