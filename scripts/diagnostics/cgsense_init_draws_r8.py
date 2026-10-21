"""Which random starting draw did each CG-SENSE LOUPE mask row start from?

`ProbMask` draws one uniform logit field per conditioning class, in class order, from the global RNG right after the
trainer seeds it, so `torch.manual_seed(seed); ProbMask(shape, slope, n)` reproduces a run's starting logits. For every
mask row of every run this reports the pixel correlation between its final logits and draws 0..2 of its seed, and the
pixel correlation between the final probability maps of all rows. A row whose final logits still correlate with its
draw has kept the draw's pixel layout; two rows from the same draw end up correlated, rows from different draws do not.

    python scripts/diagnostics/cgsense_init_draws_r8.py
"""
import json
from pathlib import Path

import torch

import _path  # noqa: F401
from activemri.samplers.loupe import ProbMask

RUNS = {
    'R8 shared 50ep conv': 'loupe2d_cgsense_r8_brain_knee_shared',
    'R8 shared 50ep conv seed43': 'loupe2d_cgsense_r8_brain_knee_shared_seed43',
    'R8 shared 50ep bilinear': 'loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep',
    'R8 shared 50ep bilinear seed43': 'loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep_seed43',
    'R8 shared 100ep bilinear': 'loupe2d_cgsense_r8_brain_knee_shared_bilinear_const_100ep',
    'R8 anatomy 50ep conv': 'loupe2d_cgsense_r8_brain_knee_anatcond',
    'R8 anatomy 100ep bilinear': 'loupe2d_cgsense_r8_brain_knee_anatcond_bilinear_const_100ep',
    'R8 protocol 50ep conv': 'loupe2d_cgsense_r8_brain_knee_prototable',
    'R8 shared continued (control)': 'loupe2d_cgsense_r8_brain_knee_shared_cont',
    'R8 anatomy shared-init': 'loupe2d_cgsense_r8_brain_knee_anatcond_sharedinit',
    'R4 shared 50ep conv': 'loupe2d_cgsense_r4_brain_knee_shared',
    'R4 anatomy 50ep conv': 'loupe2d_cgsense_r4_brain_knee_anatcond',
}
NUM_DRAWS = 3
OUT = Path('reports/cgsense_anatcond/init_draws.json')


def corr(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def seeded_draws(seed, shape, slope):
    torch.manual_seed(seed)
    return ProbMask(shape, slope, NUM_DRAWS).mask[..., 0].detach()


def main():
    rows = {}
    for run, directory in RUNS.items():
        checkpoint = torch.load(f'checkpoints/{directory}/last_model.pt', map_location='cpu', weights_only=False)
        options = checkpoint['options']
        logits = checkpoint['model']['sampler.mask_net.mask'][..., 0]
        classes = list(options.label_classes) if options.anatomy_conditioning else ['shared']
        draws = seeded_draws(options.seed, tuple(logits.shape[1:]), options.slope)
        for index, name in enumerate(classes):
            rows[f'{run} | {name}'] = {
                'seed': options.seed,
                'corr_with_draw': [round(corr(logits[index], draw), 3) for draw in draws],
                'probability': torch.sigmoid(options.slope * logits[index]),
            }

    names = list(rows)
    probabilities = torch.stack([rows[name].pop('probability').flatten() for name in names])
    pairwise = torch.corrcoef(probabilities)
    for i, name in enumerate(names):
        rows[name]['corr_with_rows'] = {other: round(pairwise[i, j].item(), 3) for j, other in enumerate(names) if j != i}

    width = max(map(len, names))
    print(f'{"row":<{width}}  seed  final-logit corr with draw 0 / 1 / 2')
    for name in names:
        print(f'{name:<{width}}  {rows[name]["seed"]:>4}  ' + ' / '.join(f'{c:+.2f}' for c in rows[name]['corr_with_draw']))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
