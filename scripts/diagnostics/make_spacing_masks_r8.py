"""Fixed R=8 masks for the spacing test: does a Poisson-disc mask beat an independent draw from the same density?

Every mask comes from the learned VarNet LOUPE density (budget-rescaled as the trainer rescales it), with the
same 10 x 10 centre and exactly 8192 samples (8092 + 100 centre):

    pd_loupe_s<k>     variable-density Poisson-disc (no two samples closer than c / sqrt(p_in)), with its input
                      density p_in iteratively compensated so the realized density matches the learned one.
                      Textbook Poisson-disc cannot hold this density: it under-samples the inner rings by up to
                      57% and pushes the samples outward, which would confound spacing with density.
    white_loupe_s<k>  independent draw: the 8092 locations with the smallest u / p, u white noise (order sampling,
                      inclusion probability ~p at an exact count)
    topk_loupe        the 8092 highest-probability locations: deterministic

The learned density is structured at the pixel scale (outside r = 96 px the median p is 0.004, the mean 0.056),
so spacing can only act between neighbouring pixels: Poisson-disc g(1) falls from ~0.9 at the centre to ~0.1
outside 96 px while g(2) stays ~1. A Poisson-disc of the same density blurred by 4 px would space samples
further, but on the grid it formed 2 px lattices (g(2) ~1.5) and could not be density-matched (worst ring off by
11-32%), so it is not used.

Each mask is checked against its density ring by ring, and its pair correlation g(d) (< 1: neighbours avoided)
is reported over the whole map and per radius band. `--verify` also checks on a real val batch that the density
and centre match the reference model's own `propose_mask` and `_initial_masks`.

    python scripts/diagnostics/make_spacing_masks_r8.py --verify
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import _path  # noqa: F401
from activemri.utils.mask_binarization import RescaleProbMap
from mask_spacing_r8 import VARNET_LOUPE, pair_correlation

OUT_DIR = Path('analysis/mask_spacing/masks_r8')
RING_WIDTH = 8
COMPENSATION_SIGMA = 6.0
BANDS = ((0, 24), (24, 48), (48, 96), (96, 182))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, default=VARNET_LOUPE / 'last_model.pt')
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--compensation-iterations', type=int, default=8)
    parser.add_argument('--verify', action='store_true', help='compare with the reference model on a val batch')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def centre_mask(shape, preselect_num):
    """The pre-selected centre block, built as SequentialSampling._init_mask does (corners, then fftshift)."""
    corners = torch.zeros(shape)
    n = preselect_num // 2
    corners[:n, :n] = corners[:n, -n:] = corners[-n:, :n] = corners[-n:, -n:] = 1
    return torch.fft.fftshift(corners)


def learned_density(checkpoint_path):
    """(density over the non-centre locations, centre, budget) of a loupe_2d checkpoint, as its forward pass rescales it."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    options = checkpoint['options']
    logits = checkpoint['model']['sampler.mask_net.mask'][0, ..., 0].float()
    shape = tuple(logits.shape)
    centre = centre_mask(shape, options.preselect_num)
    available = 1 - centre
    budget = (1 / options.accelerations[0] - options.preselect_num ** 2 / (shape[0] * shape[1])) * shape[0] * shape[1]
    probability = torch.sigmoid(options.slope * logits) * available
    density = RescaleProbMap(probability[None, None], budget / (shape[0] * shape[1]), available[None, None])[0, 0]
    return density, centre, round(budget)


def local_mean(x, available, sigma):
    """Gaussian-weighted mean of x over the available locations."""
    weight = gaussian_filter(available.astype(np.float64), sigma, mode='constant')
    return gaussian_filter(x * available, sigma, mode='constant') / np.clip(weight, 1e-6, None)


def poisson_disc(density, target, seed, iterations=14):
    """Variable-density Poisson-disc mask with exactly `target` samples on the density > 0 locations.

    Locations are visited in a density-weighted random order (keys u^(1/p)) and accepted unless an accepted one lies
    closer than c / sqrt(p); c is bisected to the largest value that still reaches the target. Returns (mask, c).
    """
    h, w = density.shape
    flat = density.ravel()
    order = np.argsort(-np.random.default_rng(seed).random(h * w) ** (1.0 / np.clip(flat, 1e-6, None)))
    order = order[flat[order] > 0]
    ys, xs = np.divmod(order, w)
    unit_radius = 1.0 / np.sqrt(np.clip(density, 1e-6, None))

    def draw(c):
        mask = np.zeros((h, w), dtype=bool)
        count = 0
        for y, x in zip(ys, xs):
            r = c * unit_radius[y, x]
            ri = int(np.ceil(r))
            y0, y1, x0, x1 = max(0, y - ri), min(h, y + ri + 1), max(0, x - ri), min(w, x + ri + 1)
            window = mask[y0:y1, x0:x1]
            if ri > 0 and window.any():
                yy, xx = np.nonzero(window)
                if np.min((yy + y0 - y) ** 2 + (xx + x0 - x) ** 2) < r * r:
                    continue
            mask[y, x] = True
            count += 1
            if count == target:
                break
        return mask, count

    low, high = 0.2, 2.0
    for _ in range(iterations):
        c = 0.5 * (low + high)
        low, high = (c, high) if draw(c)[1] == target else (low, c)
    mask, count = draw(low)
    assert count == target, f"Poisson-disc reached {count} of {target} samples"
    return mask, low


def matched_poisson_disc(density, target, seed, iterations):
    """Poisson-disc whose realized local density matches `density`: the input density is multiplied by
    local target / local realized (Gaussian, sigma COMPENSATION_SIGMA px) after each draw. Keeps the draw with
    the smallest worst-ring deviation. Returns (mask, info)."""
    available = density > 0
    target_local = local_mean(density, available, COMPENSATION_SIGMA)
    current, best = density.copy(), None
    for _ in range(iterations):
        mask, c = poisson_disc(current, target, seed)
        check = ring_check(mask, density)
        if best is None or check['max_abs_rel_dev'] < best[1]['max_abs_rel_dev']:
            best = (mask, check, c)
        realized_local = local_mean(mask.astype(np.float64), available, COMPENSATION_SIGMA)
        current = np.where(available, np.clip(current * target_local / np.clip(realized_local, 1e-4, None), 1e-4, 50.0), 0.0)
    mask, _, c = best
    return mask, {'c': round(c, 4)}


def order_sample(density, target, seed):
    """The `target` locations with the smallest u / p, u ~ U(0, 1) i.i.d.: an independent draw at an exact count."""
    ratio = np.where(density > 0, np.random.default_rng(seed).random(density.shape) / np.clip(density, 1e-12, None), np.inf)
    mask = np.zeros(density.size, dtype=bool)
    mask[np.argsort(ratio.ravel())[:target]] = True
    return mask.reshape(density.shape)


def topk(density, target):
    mask = np.zeros(density.size, dtype=bool)
    mask[np.argsort(-density.ravel(), kind='stable')[:target]] = True
    return mask.reshape(density.shape)


def _radius(shape):
    yy, xx = np.meshgrid(np.arange(shape[0]) - shape[0] // 2, np.arange(shape[1]) - shape[1] // 2, indexing='ij')
    return np.hypot(yy, xx)


def ring_check(mask, density):
    """Per-ring sample count against the density's expected count, over rings of RING_WIDTH px from the centre."""
    radius = _radius(density.shape)
    rows = []
    for inner in range(0, int(radius.max()) + 1, RING_WIDTH):
        ring = (radius >= inner) & (radius < inner + RING_WIDTH)
        expected = float(density[ring].sum())
        if expected >= 50:
            rows.append({'inner_px': inner, 'expected': round(expected, 1), 'count': int(mask[ring].sum()),
                         'rel_dev': round(float(mask[ring].sum()) / expected - 1, 4)})
    deviations = np.abs([row['rel_dev'] for row in rows])
    return {'max_abs_rel_dev': float(deviations.max()), 'mean_abs_rel_dev': float(deviations.mean()), 'rings': rows}


def band_pair_correlation(mask, density, offsets=(1, 2, 3)):
    """g(d) per radius band, pairs with both ends in the band, normalised by the density's own pair product."""
    radius = _radius(density.shape)
    h, w = density.shape
    mask = mask.astype(np.float64)
    out = {}
    for inner, outer in BANDS:
        band = (radius >= inner) & (radius < outer)
        values = []
        for d in offsets:
            ratios = []
            for dy, dx in ((0, d), (d, 0), (d, d)):
                both = band[:h - dy, :w - dx] & band[dy:, dx:]
                pairs = (mask[:h - dy, :w - dx] * mask[dy:, dx:])[both].sum()
                expected = (density[:h - dy, :w - dx] * density[dy:, dx:])[both].sum()
                ratios.append(pairs / expected)
            values.append(round(float(np.mean(ratios)), 3))
        out[f'{inner}-{outer}px'] = values
    return out


@torch.no_grad()
def verify(checkpoint_path, density, centre, device):
    """Max |difference| between this density + centre and the model's own rescaled map on a val batch, and the
    centre against `_initial_masks`, so the saved masks share the trainer's layout."""
    from omegaconf import OmegaConf
    from activemri.training.builders import build_dataloaders
    from mask_gradient_anatomy_r8 import load

    model, _ = load(checkpoint_path.parent, checkpoint_path.name, device)
    cfg = OmegaConf.load(checkpoint_path.parent / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 2
    _, val_loader = build_dataloaders(cfg)
    kspace = next(iter(val_loader))['kspace'][:4].to(device)
    proposal = model.propose_mask(kspace)
    model_map = (proposal['rescaled'] + proposal['old_mask']).clamp(max=1).cpu()
    old_mask, valid_mask = model._initial_masks(kspace)
    return {'max_abs_density_diff': float((model_map - (density + centre)).abs().max()),
            'centre_matches_initial_mask': bool((old_mask.cpu() == centre).all()),
            'invalid_locations_in_batch': int((valid_mask == 0).sum())}


def save_figure(masks, densities, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = [(f'{name} density', density) for name, density in densities.items()] + list(masks.items())
    fig, axes = plt.subplots(2, len(panels), figsize=(2.6 * len(panels), 5.6), facecolor='#fcfcfb')
    zoom = slice(80, 176)
    for col, (name, image) in enumerate(panels):
        axes[0, col].imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[0, col].set_title(name, fontsize=8)
        axes[1, col].imshow(image[zoom, zoom], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    axes[1, 0].set_ylabel('central 96 x 96', fontsize=8)
    fig.suptitle('Spacing-test masks, R=8: same density, centre and count', fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches='tight', facecolor=fig.get_facecolor())


def main():
    args = parse_args()
    density, centre, budget = learned_density(args.checkpoint)
    p = density.numpy().astype(np.float64)

    masks, info = {}, {}
    for seed in args.seeds:
        masks[f'pd_loupe_s{seed}'], info[f'pd_loupe_s{seed}'] = matched_poisson_disc(
            p, budget, seed, args.compensation_iterations)
        masks[f'white_loupe_s{seed}'], info[f'white_loupe_s{seed}'] = order_sample(p, budget, seed), {}
        print(f'seed {seed} done', flush=True)
    masks['topk_loupe'], info['topk_loupe'] = topk(p, budget), {}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {'checkpoint': str(args.checkpoint), 'non_centre_budget': budget, 'centre_samples': int(centre.sum()),
               'masks': {}}
    for name, mask in masks.items():
        full = torch.from_numpy(mask).float() + centre
        assert full.max() == 1 and int(full.sum()) == budget + int(centre.sum())
        g = pair_correlation(full, 4.0)['mean']
        info[name].update({'seed': int(name.rsplit('_s', 1)[1]) if '_s' in name else None, 'samples': int(full.sum()),
                           'g_global_d1_d2': [round(g[0], 3), round(g[1], 3)],
                           'g_by_band_d1_d2_d3': band_pair_correlation(mask, p),
                           'ring_check': ring_check(mask, p)})
        summary['masks'][name] = info[name]
        torch.save({'mask': full.to(torch.uint8), 'name': name, 'source': str(args.checkpoint),
                    **{k: v for k, v in info[name].items() if k != 'ring_check'}}, args.out_dir / f'{name}.pt')
        ring = info[name]['ring_check']
        print(f"{name:16s} samples {int(full.sum())}  ring dev max {ring['max_abs_rel_dev']:.3f} mean "
              f"{ring['mean_abs_rel_dev']:.3f}  g(1) by band {[v[0] for v in info[name]['g_by_band_d1_d2_d3'].values()]}")
    if args.verify:
        summary['verify'] = verify(args.checkpoint, density, centre, args.device)
        print(json.dumps(summary['verify']))
    (args.out_dir / 'masks_r8.json').write_text(json.dumps(summary, indent=1) + '\n')
    save_figure({name: masks[name].astype(float) + centre.numpy() for name in masks}, {'loupe': p + centre.numpy()},
                args.out_dir / 'masks_r8.png')


if __name__ == '__main__':
    main()
