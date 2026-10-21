"""Did the DCM rejection map train? CG-SENSE R=8, LOUPE vs DCM selection.

The loupe_2d logits start as sigmoid(slope * logit) ~ U(0.01, 0.99), an unstructured random field. The model is
rebuilt with the training seed to recover that init (checked on LOUPE: a reproduced init correlates with the trained
map of a shared-init conditioned run). For each run and class this reports:
  - corr(init, trained) of the unscaled probability, and the logit change relative to the init logit spread;
  - corr(trained probability, k-space radius): structure the map learned beyond the prior;
  - DCM only: spread of the score (acceptance - rescaled rejection) over available locations, the softmax allocation
    max/min ratio at tau, how many locations sit within a small margin of the top-M threshold, and Dice of the
    mask against the init mask and against best_model.pt.

    python scripts/diagnostics/dcm_training_check_r8.py
"""
import argparse
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import build_model

RUNS = {
    'LOUPE shared': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared',
    'LOUPE anatomy': 'checkpoints/loupe2d_cgsense_r8_brain_knee_anatcond',
    'DCM shared': 'checkpoints/loupe2d_cgsense_r8_brain_knee_dcm_shared',
    'DCM anatomy': 'checkpoints/loupe2d_cgsense_r8_brain_knee_dcm_anatcond',
    'DCM protocol': 'checkpoints/loupe2d_cgsense_r8_brain_knee_dcm_prototable',
}
MARGINS = (0.01, 0.05)
OUT = Path('reports/cgsense_mask_variants_r8/dcm_training_check.json')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def built(path, name, seed=None):
    checkpoint = torch.load(Path(path) / name, map_location='cpu', weights_only=False)
    if seed is not None:
        torch.manual_seed(seed)
    model = build_model(checkpoint['options'])
    if seed is None:
        model.load_state_dict(checkpoint['model'])
    return model.eval(), checkpoint['options']


def corr(a, b):
    return float(np.corrcoef(a.flatten().numpy(), b.flatten().numpy())[0, 1])


def dice(a, b):
    return float(2 * (a * b).sum() / (a.sum() + b.sum()))


@torch.no_grad()
def dcm_state(model, labels, kspace):
    """Score and soft allocation over the available locations, and the chosen mask, of a DCM model for one class."""
    unscaled = model.propose_mask(kspace, anatomy_labels=labels)['unscaled']
    old_mask, valid_mask = model._initial_masks(kspace)
    sampler = model.sampler
    available = (1 - old_mask) * valid_mask
    allocation, mask = sampler.select_locations(unscaled, old_mask, model.budget, valid_mask)
    score = sampler.dcm_tau * torch.log(allocation.clamp_min(1e-30))
    return score[available > 0], allocation[available > 0], mask[0, 0], int(round(float(model.budget)))


def main():
    args = parse_args()
    cfg = OmegaConf.load(Path(RUNS['LOUPE shared']) / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 2
    kspace = next(iter(build_dataloaders(cfg)[1]))['kspace'][:1]
    result = {}
    for run, path in RUNS.items():
        trained, options = built(path, 'last_model.pt')
        init, _ = built(path, 'last_model.pt', seed=args.seed)
        best, _ = built(path, 'best_model.pt')
        shape = tuple(trained.sampler.mask_net.mask.shape[1:3])
        radius = torch.hypot(*torch.meshgrid((torch.arange(shape[0]) - shape[0] // 2).float(),
                                             (torch.arange(shape[1]) - shape[1] // 2).float(), indexing='ij'))
        classes = list(options.label_classes) if options.anatomy_conditioning else ['all']
        result[run] = {}
        for index, name in enumerate(classes):
            logit, logit0 = (m.sampler.mask_net.mask[index, :, :, 0].detach() for m in (trained, init))
            slope = trained.sampler.mask_net.slope
            p, p0 = torch.sigmoid(slope * logit), torch.sigmoid(slope * logit0)
            row = {'corr_init_trained': corr(p0, p), 'logit_change_over_init_std': float((logit - logit0).std() / logit0.std()),
                   'corr_trained_radius': corr(p, radius), 'corr_init_radius': corr(p0, radius)}
            if getattr(options, 'selection', 'bernoulli') == 'dcm':
                labels = torch.tensor([index]) if options.anatomy_conditioning else None
                score, allocation, mask, budget = dcm_state(trained, labels, kspace)
                threshold = torch.sort(score, descending=True).values[budget - 1]
                row.update({
                    'score_std': float(score.std()), 'score_range': [float(score.min()), float(score.max())],
                    'allocation_max_over_min': float(allocation.max() / allocation.min()),
                    'near_threshold_fraction_of_budget': {str(m): float(((score - threshold).abs() < m).sum() / budget) for m in MARGINS},
                    'dice_init_mask': dice(dcm_state(init, labels, kspace)[2], mask),
                    'dice_best_mask': dice(dcm_state(best, labels, kspace)[2], mask),
                })
            result[run][name] = row
            print(run, name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
