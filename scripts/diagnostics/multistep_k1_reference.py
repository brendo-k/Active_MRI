"""Forward + backward of a small LOUPE / conv VarNet at num_step=1, saved so another checkout can be compared bitwise.

Usage: python scripts/diagnostics/multistep_k1_reference.py <checkout> <out.pt>, once for main and once for the branch,
then scripts/diagnostics/multistep_k1_compare.py <main.pt> <branch.pt>.
"""
import sys

checkout, out_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, checkout)

import torch  # noqa: E402

from activemri.models.sequential_sampling import SequentialSampling  # noqa: E402
from activemri.reconstructors.VarNet import VarNet  # noqa: E402

SIZE, COILS = 64, 4
CASES = {
    'loupe_2d': dict(sampler_type='loupe_2d', loupe_vd_init_power=4.0),
    'loupe_2d_cond': dict(sampler_type='loupe_2d', num_conditioning_classes=3),
    'conv_2d': dict(sampler_type='conv_2d', sampler_chans=8, sampler_depth=2, sampler_image_input=True),
    'conv_2d_rss': dict(sampler_type='conv_2d', sampler_chans=8, sampler_depth=2, sampler_image_input=True,
                        sampler_coil_combine='rss'),
}


def batch():
    generator = torch.Generator().manual_seed(7)
    kspace = torch.randn(2, COILS, SIZE, SIZE, 2, generator=generator)
    kspace[1, :, :, :12] = 0
    kspace[1, :, :, -12:] = 0
    kspace[1, :, :6] = 0
    target = torch.rand(2, 1, SIZE, SIZE, generator=generator)
    return kspace, target


def run(name, kwargs):
    torch.manual_seed(0)
    model = SequentialSampling(VarNet(num_cascades=2, chans=4, pools=2), shape=(SIZE, SIZE), sparsity=0.25,
                               num_step=1, preselect_num=4, recon_type='varnet', budget_over='acquired', **kwargs)
    model.train()
    kspace, target = batch()
    labels = torch.tensor([0, 2])
    torch.manual_seed(1)
    pred = model(target, kspace, anatomy_labels=labels)
    meta = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0}
    loss, _ = model.loss(pred, {'target': target, 'kspace': kspace}, meta, 'l1')
    loss.backward()
    return {
        'loss': loss.detach(),
        'output': pred['output'].detach(),
        'mask': pred['mask'].detach(),
        'grads': {key: p.grad.detach().clone() for key, p in model.named_parameters() if p.grad is not None},
    }


torch.set_num_threads(1)
torch.save({name: run(name, kwargs) for name, kwargs in CASES.items()}, out_path)
print('saved', out_path)
