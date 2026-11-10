"""How far K equal-budget steps of one static LOUPE map move its marginals from the K=1 mask (P0 item 5).

Runs the branch's own selection code (Sampler.select_locations), step by step as SequentialSampling.forward does,
on a trained loupe_2d map, and compares each location's coverage at K = 4, 16 with its K=1 rescaled probability.

Usage: python scripts/diagnostics/multistep_union_marginals.py <checkout> <loupe checkpoint> [draws]
"""
import sys

checkout, checkpoint_path = sys.argv[1], sys.argv[2]
draws = int(sys.argv[3]) if len(sys.argv) > 3 else 256
sys.path.insert(0, checkout)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from activemri.models.sequential_sampling import SequentialSampling  # noqa: E402

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
options = checkpoint['options']
logits = checkpoint['model']['sampler.mask_net.mask']
size = logits.shape[1]
BINS = [0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0001]


def model(num_step):
    sampling_model = SequentialSampling(nn.Identity(), shape=(size, size), sparsity=1 / 8, sampler_type='loupe_2d',
                                        num_step=num_step, preselect_num=10, recon_type='varnet',
                                        budget_over='acquired', slope=options.slope)
    with torch.no_grad():
        sampling_model.sampler.mask_net.mask.copy_(logits)
    return sampling_model.eval()


def coverage(num_step, valid):
    torch.manual_seed(0)
    sampling_model = model(num_step)
    sampler = sampling_model.sampler
    valid = valid.expand(draws, -1, -1, -1)
    mask = sampling_model._init_mask(torch.zeros(draws))
    step_budget = sampling_model.mask_budget(mask, valid) / num_step
    placeholder = torch.zeros(draws, size, size, 2)
    first_rescaled = None
    with torch.no_grad():
        for _ in range(num_step):
            unscaled, _, _ = sampler.unscaled_probabilities(placeholder, mask)
            rescaled, mask = sampler.select_locations(unscaled, mask, step_budget, valid)
            first_rescaled = rescaled[0] if first_rescaled is None else first_rescaled
    return mask.mean(dim=0)[0], first_rescaled[0], mask.flatten(1).sum(dim=1).float()


def report(name, valid):
    single, single_rescaled, single_counts = coverage(1, valid)
    centre = model(1)._init_mask(torch.zeros(1))[0, 0]
    reference = torch.where(centre > 0, torch.ones_like(single_rescaled), single_rescaled)
    inside = valid[0, 0] > 0
    print(f'\n== {name}: acquired {int(inside.sum())} locations, K=1 mean count {single_counts.mean():.0f}')
    header = ' '.join(f'[{lo:.2f},{hi:.2f})' for lo, hi in zip(BINS[:-1], BINS[1:]))
    print(f'K=1 marginal bins:       {header}')
    bins = [(reference >= lo) & (reference < hi) & inside for lo, hi in zip(BINS[:-1], BINS[1:])]
    print('locations per bin:      ', ' '.join(f'{int(b.sum()):>12d}' for b in bins))
    print('K=1 mean marginal:      ', ' '.join(f'{reference[b].mean():12.3f}' for b in bins))
    for num_step in (4, 16):
        covered, _, counts = coverage(num_step, valid)
        print(f'K={num_step:<2d} mean coverage:      ', ' '.join(f'{covered[b].mean():12.3f}' for b in bins),
              f'| count {counts.mean():.0f} | mean |coverage - K=1| {((covered - reference).abs()[inside]).mean():.4f}')


full = torch.ones(1, 1, size, size)
flair = torch.zeros(1, 1, size, size)
flair[..., :, (size - 204) // 2:(size - 204) // 2 + 204] = 1
report('full grid (knee-like)', full)
report('204 of 320 columns (FLAIR-like extent)', flair)
