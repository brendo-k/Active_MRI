"""Figure + smoothness numbers for the tuned PISCO maps next to ACS24 and the trained CG-SENSE net's maps."""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _path  # noqa: F401
from tune_pisco_support import clean_anatomy, net_baseline, split_dataset
from activemri.samplers.oracle import estimate_sense_maps
from activemri.utils.pisco import PRESETS, pisco_estimate

COILS = (0, 5, 10)


def pisco_estimator(anatomy):
    config = PRESETS[anatomy]
    return lambda kspace: pisco_estimate(kspace, config).with_support(config)


def interior_variation(maps, support, region):
    """Mean |grad |S||  / mean |S| over the eroded support: how much structure the maps carry inside the object."""
    interior = (-torch.nn.functional.max_pool2d(-(support * region)[None, None], 9, stride=1, padding=4))[0, 0] > 0
    magnitude = maps.abs()
    gy, gx = magnitude[:, 1:, :-1] - magnitude[:, :-1, :-1], magnitude[:, :-1, 1:] - magnitude[:, :-1, :-1]
    gradient = (gy ** 2 + gx ** 2).sqrt()
    inside = interior[:-1, :-1]
    return float(gradient[:, inside].mean() / magnitude[:, interior].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='reports/pisco_sense_maps')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    net = net_baseline(device)
    for anatomy, picks in (('brain', (0.5, 1.0)), ('knee', (0.5, 0.05))):
        dataset, position = split_dataset(anatomy, 'val', 3, 8)
        variation = {'acs24 (no support)': [], 'pisco (tuned)': [], 'net_cgsense': []}
        shown = []
        for i in range(len(dataset)):
            sample = dataset[i]
            kspace = torch.view_as_complex(sample['kspace']).to(device)
            target = sample['target'].to(device)
            region = clean_anatomy(target)
            sources = {'acs24 (no support)': estimate_sense_maps(kspace, acs=24, threshold=0.0),
                       'pisco (tuned)': pisco_estimator(anatomy)(kspace),
                       'net_cgsense': net(kspace)}
            for name, (maps, support) in sources.items():
                variation[name].append(interior_variation(maps, support, region))
            if any(abs(position[i] - p) < 0.07 for p in picks) and len(shown) < 2 and (not shown or shown[-1][0] != abs(position[i])):
                shown.append((abs(position[i]), target, sources, region))
        print(f'== {anatomy}: interior variation of |S| (lower = smoother)', {k: round(float(np.mean(v)), 4) for k, v in variation.items()})
        for tag, (_, target, sources, region) in zip(('a', 'b'), shown):
            fig, axes = plt.subplots(len(sources), len(COILS) + 2, figsize=(3 * (len(COILS) + 2), 3 * len(sources)))
            for row, (name, (maps, support)) in enumerate(sources.items()):
                axes[row, 0].imshow(target.cpu(), cmap='gray'); axes[row, 0].set_ylabel(name)
                axes[row, 1].imshow(support.cpu(), cmap='gray'); axes[row, 1].set_title('support' if row == 0 else '')
                for col, coil in enumerate(COILS):
                    axes[row, col + 2].imshow(maps[coil].abs().cpu(), vmin=0, vmax=1, cmap='viridis')
                    axes[row, col + 2].set_title(f'|S_{coil}|' if row == 0 else '')
            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout(); fig.savefig(out / f'{anatomy}_{tag}.png', dpi=70); plt.close(fig)


if __name__ == '__main__':
    main()
