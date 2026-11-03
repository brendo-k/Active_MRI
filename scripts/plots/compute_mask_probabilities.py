"""Mean per-location sampling probability of every mask, per anatomy, for one reconstructor.

LOUPE / anatomy-LOUPE: the sampler's probability after budget rescaling (pre-selected centre = 1), averaged over validation slices.
Static masks: the fixed pattern (restricted to each slice's acquired k-space for the CG masks, as plot_cg_mask_comparison.py draws
them; as the model builds them for PromptMR), averaged over the same slices. Writes `<out>` (npz: `<condition>__<method>` maps and
`<condition>__n` counts) for plot_static_vs_dynamic_masks.py.

    python scripts/plots/compute_mask_probabilities.py --which cg --device cuda
    python scripts/plots/compute_mask_probabilities.py --which promptmr --checkpoint epoch20_model.pt --device cuda
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import _path  # noqa: F401
from activemri.samplers.caipi import caipi_mask
from activemri.samplers.poisson import poisson_mask
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import REINFORCE_VAL_MASK_SEED, anatomy_labels_from_batch
from plot_cg_mask_comparison import (CALIBRATION, CONDITIONS, ROUTED_RUN, SINGLE_RUN, fixed_mask, load, middle_indices, run_config, subsample,
                                     to_device)
from plot_promptmr_mask_comparison import LOG_ROOT, RUNS, load as load_promptmr


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--which', choices=('cg', 'promptmr'), required=True)
    parser.add_argument('--checkpoint', default='epoch20_model.pt', help='PromptMR checkpoint name inside each run directory')
    parser.add_argument('--per-anatomy', type=int, default=60, help='random middle-third validation slices per dataset part')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, default=None)
    return parser.parse_args()


def loupe_probability(model, options, batch, device):
    labels = anatomy_labels_from_batch(batch, options.label_classes, device, options.label_key)
    generator = torch.Generator(device=device)
    generator.manual_seed(REINFORCE_VAL_MASK_SEED)
    model.sampler.eval_generator = generator
    with torch.no_grad():
        out = model.propose_mask(batch['kspace'], anatomy_labels=labels)
    return torch.clamp(out['old_mask'] + out['rescaled'], 0, 1)


def build(args):
    """(probability functions by method, key -> callable(batch) -> [B,1,H,W]), plus the run whose config builds the loader."""
    if args.which == 'cg':
        loupe, loupe_options = load(SINGLE_RUN, args.device)
        anat, anat_options = load(ROUTED_RUN, args.device)
        patterns = {}

        def static(name):
            def fn(batch):
                if not patterns:
                    shape = tuple(batch['kspace'].shape[-3:-1])
                    patterns['pi'] = caipi_mask(shape, accel_y=2, accel_z=2, shift=0)
                    patterns['poisson'] = poisson_mask(shape, 4.0, CALIBRATION)
                return fixed_mask(loupe, batch['kspace'], patterns[name].to(args.device))
            return fn
        functions = {'pi': static('pi'), 'poisson': static('poisson')}
        run = SINGLE_RUN
    else:
        models = {name: load_promptmr(run_name, args.checkpoint, args.device)[:2] for name, run_name in RUNS.items()}
        loupe, loupe_options = models['loupe']
        anat, anat_options = models['loupe_anat']

        def static(name, attribute):
            model = models[name][0]

            def fn(batch):
                old_mask, _ = model._initial_masks(batch['kspace'])
                return torch.clamp(old_mask + getattr(model.sampler, attribute).to(old_mask.dtype), max=1)
            return fn
        functions = {'pi': static('pi', 'caipi_pattern'), 'poisson': static('poisson', 'poisson_pattern')}
        run = LOG_ROOT / RUNS['loupe']
    functions['loupe'] = lambda batch: loupe_probability(loupe, loupe_options, batch, args.device)
    functions['loupe_anat'] = lambda batch: loupe_probability(anat, anat_options, batch, args.device)
    return functions, run


def main():
    args = parse_args()
    out = args.out or Path('reports/mask_probabilities') / f'{args.which}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    functions, run = build(args)
    _, loader = build_dataloaders(run_config(run, args.num_workers))
    indices = subsample(loader.dataset, middle_indices(loader.dataset, 1 / 3), args.per_anatomy)
    subset = DataLoader(Subset(loader.dataset, indices), batch_size=args.batch_size, collate_fn=loader.collate_fn)
    totals, counts = {}, defaultdict(int)
    for number, batch in enumerate(subset):
        batch = to_device(batch, args.device) | {k: v for k, v in batch.items() if k not in ('kspace', 'target')}
        try:
            maps = {name: fn(batch).float().cpu() for name, fn in functions.items()}
        except RuntimeError as err:
            if 'Binarization failed' not in str(err):
                raise
            continue
        for i, condition in enumerate(batch['condition']):
            if condition not in CONDITIONS:
                continue
            counts[condition] += 1
            for name, value in maps.items():
                key = f'{condition}__{name}'
                totals[key] = totals.get(key, 0) + value[i, 0].numpy()
        print(f'batch {number + 1}/{len(subset)}', flush=True)
    np.savez_compressed(out, **{key: total / counts[key.split('__')[0]] for key, total in totals.items()},
                        **{f'{c}__n': np.array(counts[c]) for c in CONDITIONS})
    print({c: counts[c] for c in CONDITIONS}, '->', out)


if __name__ == '__main__':
    main()
