"""Which steps' probabilities get gradient at K=4 in a checkout (loupe_2d), and how many reconstructions run.

Usage: python scripts/diagnostics/multistep_step_gradients.py <checkout>
"""
import sys
sys.path.insert(0, sys.argv[1])
from types import SimpleNamespace
import torch
from torch import nn
from activemri.models.sequential_sampling import SequentialSampling
from activemri.utils.image_processing import ifft_2d_img

class Recon(nn.Module):
    def __init__(self):
        super().__init__(); self.scale = nn.Parameter(torch.ones(())); self.calls = 0
    def forward(self, k, maps, mask, **kw):
        self.calls += 1
        return self.scale * (ifft_2d_img(k) * maps.conj()).sum(dim=1, keepdim=True)

torch.manual_seed(0)
m = SequentialSampling(Recon(), shape=(64, 64), sparsity=0.25, sampler_type='loupe_2d', num_step=4, preselect_num=4,
                       recon_type='varnet', budget_over='acquired')
steps = []
select = m.sampler.select_locations
def rec(u, old, budget, valid=None):
    u.retain_grad(); steps.append(u); return select(u, old, budget, valid)
m.sampler.select_locations = rec
g = torch.Generator().manual_seed(0)
k = torch.randn(2, 4, 64, 64, 2, generator=g)
maps = torch.randn(2, 4, 64, 64, dtype=torch.complex64, generator=g); maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
t = torch.rand(2, 1, 64, 64, generator=g)
out = m(t, k, sense_maps=maps)
(out['output'] - t).abs().mean().backward()
print('reconstructions per forward:', m.reconstructor.calls)
print('steps with gradient:', [i for i, u in enumerate(steps) if u.grad is not None and u.grad.abs().sum() > 0])
