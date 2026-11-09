"""Figure of the three image-space masks (metric, Omega = sense-net support, M = ESPIRiT support) on val slices.

Per condition: a typical slice (median share of the metric mask inside Omega) and the worst one. Rows: the RSS target
with the mask outlines; the same at a 0-0.3 window, which shows the RSS noise floor; the nested regions, with the
part of the metric mask outside Omega (scored, but the output is 0 there) in red.

    python reports/loss_domain/diag/plot_masks.py --out reports/loss_domain/diag/masks.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, 'scripts/eval'); sys.path.insert(0, '.')
import eval_loss_domain as e  # noqa: E402
from activemri.datasets.fastmri_data import build_manifest_dataloaders  # noqa: E402
from activemri.utils.image_processing import ifft_2d_img  # noqa: E402
from activemri.utils.metric_mask import metric_support_mask  # noqa: E402

CONDITIONS = ('brain', 'knee-PD', 'knee-PDFS')
MASKS = {  # name: (colour, line style, label)
    'metric': ('#3987e5', '-', 'metric mask (scoring)'),
    'Omega': ('#d95926', '--', 'Ω: sense-net support (maps = 0 outside)'),
    'M': ('#199e70', ':', 'M: ESPIRiT support |x_E| > 0'),
}
SLIVER = '#d03b3b'
REGIONS = ListedColormap(['#2b2b29', '#199e70', '#d95926', '#3987e5', SLIVER])  # outside M, M\Ω, Ω\metric, metric∩Ω, metric\Ω


def omega(k):
    centre = torch.zeros_like(k)
    h, w = k.shape[-2] // 2, k.shape[-1] // 2
    centre[..., h - 5:h + 5, w - 5:w + 5] = 1
    rss = ifft_2d_img(k * centre).abs().pow(2).sum(1).sqrt()
    return rss > 0.1 * rss.amax(dim=(-2, -1), keepdim=True)


def collect():
    _, val_loader = build_manifest_dataloaders(manifest='configs/datasets/fastmri_local_v1/manifest.csv', roots=e.ROOTS,
        classes=['FLAIR', 'knee'], image_shape=[320, 320], batch_size=16, num_coils=20, num_workers=8,
        complex_cache_dir=e.CACHE)
    n = len(val_loader.dataset)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_loader.dataset, list(range(3, n, n // 120))),
                                         batch_size=8, num_workers=16, collate_fn=val_loader.collate_fn)
    slices = []
    with torch.no_grad():
        for batch in loader:
            k = torch.view_as_complex(batch['kspace'].contiguous())
            metric = metric_support_mask(k, list(batch['anatomy']))[:, 0].bool()
            espirit = torch.view_as_complex(batch['complex_target'].contiguous()).abs() > 0
            support = omega(k)
            for i, condition in enumerate(batch['condition']):
                slices.append({'condition': condition, 'name': f"{batch['fname'][i][:-3]} s{int(batch['slice_id'][i])}",
                               'target': batch['target'][i].numpy(), 'metric': metric[i].numpy(),
                               'Omega': support[i].numpy(), 'M': espirit[i].numpy(),
                               'metric_in_omega': (metric[i] & support[i]).sum().item() / metric[i].sum().item()})
    return slices


def pick(slices):
    chosen = []
    for condition in CONDITIONS:
        group = sorted((s for s in slices if s['condition'] == condition), key=lambda s: s['metric_in_omega'])
        chosen += [(group[len(group) // 2], 'typical'), (group[0], 'worst')]
    return chosen


def regions(s):
    label = np.zeros(s['M'].shape, dtype=int)
    label[s['M']] = 1
    label[s['Omega']] = 2
    label[s['metric'] & s['Omega']] = 3
    label[s['metric'] & ~s['Omega']] = 4
    return label


def outline(ax, s):
    for name, (colour, style, _) in MASKS.items():
        ax.contour(s[name].astype(float), levels=[0.5], colors=[colour], linestyles=[style], linewidths=1.6)


def plot(chosen, out):
    fig, axes = plt.subplots(3, len(chosen), figsize=(2.55 * len(chosen), 8.6), facecolor='white')
    for col, (s, kind) in enumerate(chosen):
        peak = np.percentile(s['target'][s['metric']], 99)
        image = s['target'] / peak
        axes[0, col].imshow(image, cmap='gray', vmin=0, vmax=1)
        outline(axes[0, col], s)
        axes[1, col].imshow(image, cmap='gray', vmin=0, vmax=0.3)
        outline(axes[1, col], s)
        axes[2, col].imshow(regions(s), cmap=REGIONS, vmin=-0.5, vmax=4.5, interpolation='nearest')
        axes[0, col].set_title(f"{s['condition']}, {kind}\n{s['name']}", fontsize=8.5, color='#0b0b0b')
        share = {name: s[name].mean() for name in MASKS}
        axes[2, col].set_xlabel(f"grid: metric {share['metric']:.0%}, Ω {share['Omega']:.0%}, M {share['M']:.0%}\n"
                                f"metric inside Ω {s['metric_in_omega']:.1%}", fontsize=7.5, color='#52514e')
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for row, text in enumerate(('RSS target\n(window 0-1)', 'RSS target\n(window 0-0.3: noise floor)', 'regions')):
        axes[row, 0].set_ylabel(text, fontsize=9, color='#0b0b0b')
    handles = [Line2D([], [], color=c, linestyle=st, linewidth=1.8, label=lab) for c, st, lab in MASKS.values()]
    handles += [Patch(color='#3987e5', label='metric ∩ Ω'), Patch(color='#d95926', label='Ω \\ metric'),
                Patch(color='#199e70', label='M \\ Ω'), Patch(color='#2b2b29', label='outside M'),
                Patch(color=SLIVER, label='metric \\ Ω: scored, output = 0')]
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=Path('reports/loss_domain/diag/masks.png'))
    args = parser.parse_args()
    plot(pick(collect()), args.out)


if __name__ == '__main__':
    main()
