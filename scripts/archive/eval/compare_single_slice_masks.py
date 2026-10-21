"""Compare the LOUPE patterns learned by single-slice runs.

Reads the run directories written by train_single_slice_loupe.py and answers two
questions: does the pattern change with anatomy (brain vs knee), and does it
change between slices of the same volume? A same-slice/different-seed pair is
included as the noise floor — two runs differ at least this much for reasons
that have nothing to do with the data.

    python scripts/eval/compare_single_slice_masks.py --run-dir checkpoints/single_slice_loupe \
        --suffix 2dd --out checkpoints/single_slice_loupe/compare_2d.png
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--suffix', required=True, help='run-name suffix identifying the sampler family')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smooth-sigma', type=float, default=4.0,
                        help='Gaussian sigma used to extract the density envelope of a 2D pattern. '
                             'The per-location values saturate to 0/1 and are close to random within '
                             'the envelope, so the envelope is what carries comparable structure.')
    return parser.parse_args()


def preselect_mask(shape, preselect_num, line_constrained):
    """The center region the model acquires unconditionally, in DC-centered coords.

    These locations get no gradient, so they stay at their initialization and
    must be excluded before two runs are compared.
    """
    a = np.zeros(shape)
    n = preselect_num // 2
    if line_constrained:
        a[:, :n] = 1
        a[:, -n:] = 1
    else:
        a[:n, :n] = a[:n, -n:] = a[-n:, :n] = a[-n:, -n:] = 1
    return np.fft.fftshift(a) > 0


def load_runs(run_dir, suffix):
    runs = {}
    for meta_path in sorted(run_dir.glob(f'*_{suffix}/metadata.json')):
        d = meta_path.parent
        meta = json.loads(meta_path.read_text())
        runs[d.name[: -len(suffix) - 1]] = {
            'meta': meta,
            'prob': np.load(d / 'learned_prob.npy'),
            'rescaled': np.load(d / 'rescaled_mask.npy'),
            'binary': np.load(d / 'binary_mask.npy'),
            'target': np.load(d / 'target.npy'),
            'recon': np.load(d / 'recon.npy'),
        }
    if not runs:
        raise FileNotFoundError(f'No runs matching *_{suffix} under {run_dir}')
    return runs


def compare(prob_a, prob_b, free):
    """Correlation and top-k agreement between two learned patterns."""
    a, b = prob_a[free], prob_b[free]
    corr = float(np.corrcoef(a, b)[0, 1])
    spearman = float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])

    k = max(1, int(round(0.25 * free.sum())))
    top_a = set(np.argsort(-a)[:k].tolist())
    top_b = set(np.argsort(-b)[:k].tolist())
    overlap = len(top_a & top_b) / k
    return {'pearson': corr, 'spearman': spearman, f'top{k}_overlap': overlap, 'k': k}


def envelope(prob, sigma):
    """Local sampling density: what fraction of a neighbourhood the pattern acquires."""
    return gaussian_filter(prob, sigma=sigma, mode='wrap')


def angular_profile(prob, nbins=36, rmin=8, rmax=None):
    """Mean probability by direction in k-space, over an annulus around DC.

    Two patterns with identical radial profiles can still differ in which
    directions they spend budget on; this is the statistic that shows it.
    """
    h, w = prob.shape
    yy, xx = np.meshgrid(np.arange(h) - h // 2, np.arange(w) - w // 2, indexing='ij')
    r = np.sqrt(yy ** 2 + xx ** 2)
    rmax = rmax if rmax is not None else min(h, w) // 2
    ring = (r >= rmin) & (r < rmax)
    theta = np.mod(np.arctan2(yy, xx), np.pi)          # orientation, not direction
    idx = np.minimum((theta / np.pi * nbins).astype(int), nbins - 1)
    sums = np.bincount(idx[ring], weights=prob[ring], minlength=nbins)
    counts = np.bincount(idx[ring], minlength=nbins)
    return sums / np.maximum(counts, 1)


def radial_profile(prob):
    """Mean probability as a function of distance from DC.

    The mask arrays are already DC-centered, matching the k-space convention of
    the data, so no fftshift is applied here or in the plots.
    """
    h, w = prob.shape
    yy, xx = np.meshgrid(np.arange(h) - h // 2, np.arange(w) - w // 2, indexing='ij')
    r = np.round(np.sqrt(yy ** 2 + xx ** 2)).astype(int)
    nbins = r.max() + 1
    sums = np.bincount(r.ravel(), weights=prob.ravel(), minlength=nbins)
    counts = np.bincount(r.ravel(), minlength=nbins)
    return sums / np.maximum(counts, 1)


def main():
    args = parse_args()
    runs = load_runs(args.run_dir, args.suffix)
    names = list(runs)
    any_meta = runs[names[0]]['meta']
    line_constrained = any_meta['sampler_type'] == 'loupe_1d'

    probs = {n: runs[n]['prob'] for n in names}
    shape = any_meta['resolution']
    free = ~preselect_mask(shape, any_meta['preselect_num'], line_constrained)
    if line_constrained:
        free = free[0]  # a 1D run learns one weight per column

    envelopes = ({n: envelope(probs[n], args.smooth_sigma) for n in names}
                 if not line_constrained else None)

    stats = {}
    for a, b in combinations(names, 2):
        entry = compare(probs[a], probs[b], free)
        if envelopes is not None:
            entry['envelope_pearson'] = float(
                np.corrcoef(envelopes[a][free], envelopes[b][free])[0, 1]
            )
        stats[f'{a} vs {b}'] = entry

    ncols = len(names)
    nrows = 3 if not line_constrained else 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)

    for col, name in enumerate(names):
        run = runs[name]
        meta = run['meta']
        ax[0][col].imshow(run['target'], cmap='gray')
        ax[0][col].set_title(f"{name}\nPSNR={meta['final_psnr']:.1f} SSIM={meta['final_ssim']:.3f}")
        ax[0][col].axis('off')

        if line_constrained:
            ax[1][col].plot(run['prob'])
            ax[1][col].set_ylim(0, 1)
            ax[1][col].set_title('learned line probability')
            ax[1][col].set_xlabel('phase encode (DC at center)')
        else:
            im = ax[1][col].imshow(run['prob'], cmap='inferno', vmin=0, vmax=1)
            ax[1][col].set_title('learned probability (DC centered)')
            ax[1][col].axis('off')
            fig.colorbar(im, ax=ax[1][col], fraction=0.046)
            ax[2][col].imshow(run['binary'], cmap='gray')
            ax[2][col].set_title(f"binary mask (sparsity={meta['mask_sparsity']:.3f})")
            ax[2][col].axis('off')

    fig.suptitle(f"Zero-step LOUPE, single-slice overfit — {any_meta['sampler_type']} "
                 f"R={any_meta['acceleration']:g}, {any_meta['recon_type']}", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in names:
        if line_constrained:
            ax.plot(probs[name], label=name, alpha=0.8)
            ax.set_xlabel('phase encode (DC centered)')
            ax.set_ylabel('probability')
        else:
            ax.plot(radial_profile(probs[name]), label=name, alpha=0.8)
            ax.set_xlabel('|k| (pixels from DC)')
            ax.set_ylabel('mean probability')
    ax.legend()
    ax.set_title('Learned pattern profile')
    fig.tight_layout()
    profile_path = args.out.with_name(args.out.stem + '_profile.png')
    fig.savefig(profile_path, dpi=150)
    plt.close(fig)

    if envelopes is not None:
        fig, ax = plt.subplots(2, ncols, figsize=(4.2 * ncols, 7.6), squeeze=False)
        vmax = max(e.max() for e in envelopes.values())
        for col, name in enumerate(names):
            im = ax[0][col].imshow(envelopes[name], cmap='inferno', vmin=0, vmax=vmax)
            ax[0][col].set_title(f'{name}\nenvelope (sigma={args.smooth_sigma:g})')
            ax[0][col].axis('off')
            fig.colorbar(im, ax=ax[0][col], fraction=0.046)
        gs = ax[1][0].get_gridspec()
        for a in ax[1]:
            a.remove()
        big = fig.add_subplot(gs[1, :])
        for name in names:
            big.plot(np.degrees(np.linspace(0, np.pi, 36, endpoint=False)),
                     angular_profile(probs[name]), label=name, alpha=0.85)
        big.set_xlabel('k-space orientation (degrees)')
        big.set_ylabel('mean probability')
        big.set_title('Angular profile — where the budget goes by direction')
        big.legend()
        fig.tight_layout()
        env_path = args.out.with_name(args.out.stem + '_envelope.png')
        fig.savefig(env_path, dpi=150)
        plt.close(fig)
        print(f'Wrote {env_path}')

    summary = {'runs': {n: runs[n]['meta'] for n in names}, 'pairwise': stats}
    args.out.with_suffix('.json').write_text(json.dumps(summary, indent=2))

    print(f'Wrote {args.out}, {profile_path}, {args.out.with_suffix(".json")}\n')
    width = max(len(k) for k in stats)
    for pair, s in stats.items():
        overlap_key = next(k for k in s if k.endswith('_overlap'))
        line = (f'{pair:<{width}}  pearson={s["pearson"]:+.3f}  spearman={s["spearman"]:+.3f}  '
                f'{overlap_key}={s[overlap_key]:.3f}')
        if 'envelope_pearson' in s:
            line += f'  envelope={s["envelope_pearson"]:+.3f}'
        print(line)


if __name__ == '__main__':
    main()
