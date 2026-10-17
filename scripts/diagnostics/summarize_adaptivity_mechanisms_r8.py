"""Tables and figures for reports/ADAPTIVITY_MECHANISMS_R8.md from the outputs of

  mask_gradient_anatomy_r8.py     reports/adaptivity_mechanisms_r8/gradients_<tag>.json (+ saved gradients in OUT_DIR)
  recon_bias_slice_masks_r8.py    reports/adaptivity_mechanisms_r8/slice_masks_<tag>.json
  recon_mask_tolerance_r8.py      reports/adaptivity_mechanisms_r8/tolerance.json, finetune_<recon>_sigma<s>.json
  toy_adaptive_sampling.py        <toy dir>/<arm>/history.json

Also computes the location-level pull of each reconstructor at the conv operating point: the correlation (outside
the centre) of its population descent direction on the mask, -mean_i dL/dm_i, with p_LOUPE - p_conv.
Missing inputs are skipped.

    python scripts/diagnostics/summarize_adaptivity_mechanisms_r8.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import _path  # noqa: F401
from mask_gradient_anatomy_r8 import OUT_DIR, REPORT_DIR

TOY_DIR = Path('/tmp/bkadota_toy')
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
GRADIENT_TAGS = {  # tag: (operating point, reconstructor)
    'conv_last': ('conv ep50', 'conv ep50 (co-adapted)'),
    'conv_ep4': ('conv ep4', 'conv ep4 (co-adapted)'),
    'loupe_last': ('LOUPE', 'LOUPE (co-adapted)'),
    'conv_last_recon_loupe': ('conv ep50', 'LOUPE'),
    'conv_last_recon_ep4': ('conv ep50', 'conv ep4'),
    'conv_last_recon_random': ('conv ep50', 'uniform-random-mask'),
    'loupe_last_recon_conv': ('LOUPE', 'conv ep50'),
}
SLICE_MASK_TAGS = {'conv_last': 'conv ep50 (co-adapted)', 'loupe_last': 'LOUPE', 'conv_ep4': 'conv ep4',
                   'random': 'uniform-random-mask'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--toy-dir', type=Path, default=TOY_DIR)
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def fmt(value, digits=3):
    return f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def ci(entry, digits=2):
    return f"{entry['mean']:+.{digits}f} [{entry['ci95'][0]:+.{digits}f}, {entry['ci95'][1]:+.{digits}f}]"


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


PULLS = {  # tag: sign of (p_LOUPE - p_conv) that points from the operating point toward the other sampler's masks
    'conv_last': 1, 'conv_last_recon_loupe': 1, 'conv_last_recon_ep4': 1, 'conv_last_recon_random': 1,
    'loupe_last': -1, 'loupe_last_recon_conv': -1,
}


def pull_toward_other():
    """corr(-gbar_m, p_other - p_own) outside the centre: does a reconstructor's population descent direction on the
    mask point from the operating point's masks toward the other sampler's (conv ep50 vs LOUPE)?"""
    loupe_path, conv_path = OUT_DIR / 'gradients_loupe_last.pt', OUT_DIR / 'gradients_conv_last.pt'
    if not (loupe_path.exists() and conv_path.exists()):
        return {}
    loupe, conv = torch.load(loupe_path, weights_only=False), torch.load(conv_path, weights_only=False)
    outside = ~conv['centre'][0]
    loupe_minus_conv = (loupe['p_eff'].mean(0) - conv['p_eff'].mean(0))[outside]
    out = {}
    for tag, sign in PULLS.items():
        path = OUT_DIR / f'gradients_{tag}.pt'
        if not path.exists():
            continue
        data = torch.load(path, weights_only=False)
        descent = -((data['m_a'] + data['m_b']) / 2 * ~data['centre']).mean(0)[outside]
        descent, direction = descent - descent.mean(), sign * (loupe_minus_conv - loupe_minus_conv.mean())
        out[tag] = float((descent * direction).sum() / (descent.norm() * direction.norm()))
    return out


def gradient_section(lines, summary):
    rows, scale_rows, results = [], [], {}
    for tag, (point, recon) in GRADIENT_TAGS.items():
        d = load_json(REPORT_DIR / f'gradients_{tag}.json')
        if d is None:
            continue
        results[tag] = d
        z, m, share = d['z_energies'], d['m_energies'], d['z_sharing']
        rows.append([point, recon, fmt(d['mean_loss'], 4), fmt(z['pop_frac']), fmt(z['anat_frac']), fmt(z['slice_frac']),
                     fmt(z['slice_snr_per_draw'], 2), fmt(z['anat_snr_per_draw'], 4), fmt(z['pop_snr_batch16'], 2),
                     fmt(share['split_half_reliability_16_draws'], 2), fmt(share['neighbour_corr_disattenuated_median'], 2),
                     fmt(share['other_volume_corr_disattenuated_median'], 2),
                     fmt(m['slice_frac']), fmt(d['m_region_shares']['dead']['pop_share'], 2)])
        if 'z_scales' in d:
            scale_rows.append([point, recon] + [f"{r['energy_kept']:.3f} / {r['neighbour_corr_disattenuated_median']:.2f}"
                                                for r in d['z_scales']['slice']])
    lines.append('## Gradient anatomy (z = score map; m = binary mask, outside the centre)\n')
    lines.append(table(['operating point', 'reconstructor', 'L1 loss', 'pop', 'anat', 'slice', 'slice SNR/draw',
                        'anat SNR/draw', 'pop SNR @16', 'reliability (16 draws)', 'neighbour corr', 'other-volume corr',
                        'm: slice', 'm: pop share in dead region'], rows))
    lines.append('\nSlice-specific z energy kept at scales >= sigma / neighbour corr of the blurred signal:\n')
    lines.append(table(['operating point', 'reconstructor', 'sigma 1 px', 'sigma 2 px', 'sigma 4 px', 'sigma 8 px'], scale_rows))
    pull = pull_toward_other()
    if pull:
        lines.append('\nPull toward the other sampler, corr(-mean_i dL/dm_i, p_other - p_own) outside the centre:\n')
        lines.append(table(['operating point (own masks)', 'reconstructor', 'corr'],
                           [[GRADIENT_TAGS[t][0], GRADIENT_TAGS[t][1], fmt(v)] for t, v in pull.items()]))
    summary['pull_toward_other'] = pull

    transport_rows = []
    for tag in ('conv_ep4', 'conv_last'):
        d = results.get(tag, {}).get('transport')
        if d is None:
            continue
        full_only = load_json(REPORT_DIR / f'gradients_{tag}_full_update.json') or {}
        for key in ('adam', 'sgd'):
            push = d[f'{key}_single_push']
            full = full_only.get('transport', {}).get(f'{key}_full_update', d[f'{key}_full_update'])
            controls = d.get(f'{key}_single_push_controls', {})
            brain, knee = d[f'{key}_anatomy_push_brain'], d[f'{key}_anatomy_push_knee']
            transport_rows.append([GRADIENT_TAGS[tag][0], key,
                                   fmt(full.get('common_alignment_with_population', float('nan'))),
                                   fmt(full.get('anatomy_offset_alignment', float('nan'))),
                                   fmt(push['self_cosine']),
                                   fmt(push.get('self_cosine_with_blurred_half_b_sigma4', float('nan'))),
                                   fmt(controls.get('pop', {}).get('self_cosine_with_push', float('nan'))),
                                   fmt(controls.get('anat', {}).get('self_cosine_with_push', float('nan'))),
                                   fmt(full['free_map_alignment'], 2), fmt(full['slice_specific_alignment']),
                                   fmt(push['neighbour'] / push['self'], 2), fmt(push['other'] / push['self'], 2),
                                   fmt(push['other_mean_change_over_self_change_norm'], 2),
                                   fmt(brain['response_on_knee'] / brain['response_on_brain'], 2),
                                   fmt(knee['response_on_brain'] / knee['response_on_knee'], 2)])
    if transport_rows:
        lines.append('\n## Transport through the conv sampler network\n')
        lines.append(table(['sampler', 'update', 'full update: common vs population', 'full update: anatomy offset',
                            'self cosine', 'self cosine vs blurred (4 px) target',
                            'control: population push cosine', 'control: anatomy push cosine',
                            'free-map alignment', 'net alignment (full update)', 'neighbour/self response',
                            'other/self response', '|other-volume change| / |own change|',
                            'brain push: knee/brain', 'knee push: brain/knee'], transport_rows))
    summary['gradients'] = {t: {k: v for k, v in d.items() if k != 'radial'} for t, d in results.items()}
    return results


def plot_scales(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for color, (tag, (point, recon)) in zip(COLORS, GRADIENT_TAGS.items()):
        d = results.get(tag)
        if d is None or 'z_scales' not in d:
            continue
        rows = d['z_scales']['slice']
        sigmas = [0] + [r['sigma_px'] for r in rows]
        kept = [1.0] + [r['energy_kept'] for r in rows]
        sharing = [d['z_sharing']['neighbour_corr_disattenuated_median']] + [r['neighbour_corr_disattenuated_median'] for r in rows]
        label = f'{point} masks, {recon} recon'
        axes[0].plot(sigmas, kept, '-o', color=color, lw=2, ms=4, label=label)
        axes[1].plot(sigmas, sharing, '-o', color=color, lw=2, ms=4, label=label)
    axes[0].set(xlabel='blur sigma (px in k-space)', ylabel='slice-specific energy kept', yscale='log',
                title='Scale of the per-slice gradient')
    axes[1].set(xlabel='blur sigma (px in k-space)', ylabel='neighbour corr (disattenuated, median)',
                title='Shared with the adjacent slice', ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.3, lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)
    axes[1].legend(fontsize=7, frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def radial_plot(results, path):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    d = results.get('conv_last')
    if d is None:
        return
    centres = [(r['r_low'] + r['r_high']) / 2 for r in d['radial']]
    for color, (key, label) in zip(COLORS, (('z_pop', 'population'), ('z_anat', 'anatomy offset'),
                                            ('z_slice', 'slice-specific'), ('z_noise_per_draw', 'draw noise (1 draw)'))):
        ring_area = [np.pi * (r['r_high'] ** 2 - r['r_low'] ** 2) for r in d['radial']]
        ax.plot(centres, [r[key] / a for r, a in zip(d['radial'], ring_area)], '-o', color=color, lw=2, ms=4, label=label)
    ax.set(xlabel='k-space radius (px)', ylabel='energy per location', yscale='log',
           title='conv ep50: score-gradient energy by radius')
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def slice_mask_section(lines, summary):
    rows = []
    for tag, recon in SLICE_MASK_TAGS.items():
        d = load_json(REPORT_DIR / f'slice_masks_{tag}.json')
        if d is None:
            continue
        summary.setdefault('slice_masks', {})[tag] = d
        rows.append([recon, fmt(d['own']['psnr_shared'], 2), ci(d['own']['shared_minus_prior_db']),
                     ci(d['own']['slice_minus_shared_db']), ci(d['neighbour']['slice_minus_shared_db']),
                     ci(d['other']['slice_minus_shared_db']), fmt(d['masks']['slice_vs_shared_corr'], 3),
                     fmt(d['masks']['neighbour_deviation_corr'], 3), fmt(d['masks']['other_deviation_corr'], 3)])
    if rows:
        lines.append('\n## Per-slice free masks fitted through each frozen reconstructor (gain over its shared refit, dB)\n')
        lines.append(table(['reconstructor', 'shared PSNR', 'shared - prior', 'own slice', 'neighbour slice',
                            'other volumes', 'slice vs shared map corr', 'neighbour deviation corr',
                            'other-volume deviation corr'], rows))


def tolerance_section(lines, summary):
    d = load_json(REPORT_DIR / 'tolerance.json')
    if d is not None:
        summary['tolerance'] = d
        sigmas = [s for s in d['config']['sigmas'] if s > 0]
        fractions = d['config']['ring_fractions']
        rows = []
        for name in ('conv_last', 'conv_ep4', 'loupe_last', 'random'):
            if name not in d:
                continue
            r = d[name]
            rows.append([name, fmt(r['base_psnr'], 2)]
                        + [f"{ci(r[f'reshape_{s:g}_0_minus_base_db'])} (r {r['reshape_map_corr'][f'{s:g}']:.2f})" for s in sigmas]
                        + [ci(r[f'ring_0_{f:g}_minus_base_db']) for f in fractions])
        lines.append('\n## Tolerance to mask deviations (PSNR change vs the reconstructor\'s base masks, dB; r = map corr)\n')
        lines.append(table(['reconstructor', 'base PSNR'] + [f'reshape sigma {s:g}' for s in sigmas]
                           + [f'ring {f:g}' for f in fractions], rows))
        plot_tolerance(d, REPORT_DIR / 'tolerance.png')
    rows = []
    for path in sorted(REPORT_DIR.glob('finetune_*.json')):
        f = json.loads(path.read_text())
        summary.setdefault('finetune', {})[path.stem] = f
        rows.append([f['config']['recon'], f['config']['sigma'], f['config']['steps'], ci(f['penalty_frozen_db']),
                     ci(f['penalty_after_adapting_db']), ci(f['mismatch_db']), ci(f['ft_perturbed_cost_on_unperturbed_db'])])
    if rows:
        lines.append('\n## Fine-tune control: mismatch vs information\n')
        lines.append(table(['reconstructor', 'reshape sigma', 'fine-tune steps', 'penalty, frozen',
                            'penalty after adapting', 'mismatch (frozen - adapted)', 'perturbed-trained cost on base masks'], rows))


def plot_tolerance(d, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sigmas = d['config']['sigmas']
    fractions = d['config']['ring_fractions']
    for color, name in zip(COLORS, ('conv_last', 'conv_ep4', 'loupe_last', 'random')):
        if name not in d:
            continue
        r = d[name]
        corr = [r['reshape_map_corr'][f'{s:g}'] for s in sigmas]
        delta = [0.0] + [r[f'reshape_{s:g}_0_minus_base_db']['mean'] for s in sigmas[1:]]
        axes[0].plot(corr, delta, '-o', color=color, lw=2, ms=4, label=name)
        axes[1].plot([0] + list(fractions), [0.0] + [r[f'ring_0_{f:g}_minus_base_db']['mean'] for f in fractions],
                     '-o', color=color, lw=2, ms=4, label=name)
    axes[0].set(xlabel='correlation of perturbed with base probability map', ylabel='PSNR change (dB)',
                title='Smooth per-slice density reshape')
    axes[0].invert_xaxis()
    axes[1].set(xlabel='fraction of samples moved within their ring', ylabel='PSNR change (dB)',
                title='Same radial density, new arrangement')
    for ax in axes:
        ax.grid(alpha=0.3, lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


TOY_GROUPS = {
    'A_frozen_gen': 'frozen generalist recon',
    'B_joint_scratch': 'joint from scratch',
    'C_lockin_coadapted': 'joint, from co-adapted static recon + mask',
    'C_lockin_coadapted_frozen': 'frozen co-adapted static recon, from its mask',
    'D_from_static_generalist_joint': 'joint, generalist recon, from the static mask',
    'D_from_static_generalist_frozen': 'frozen generalist recon, from the static mask',
}


def toy_runs(toy_dir):
    runs = {}
    for path in sorted(toy_dir.glob('*/history.json')):
        d = json.loads(path.read_text())
        if 'final' in d:
            runs[path.parent.name] = d
    return runs


@torch.no_grad()
def toy_determinism(run_dir, config):
    """Share of non-centre locations with effective probability in (0.05, 0.95) on the toy validation images."""
    import toy_adaptive_sampling as toy
    sampler = (toy.Net(net_input=config.get('net_input', 'raw')) if config['sampler'] == 'net'
               else toy.SAMPLERS[config['sampler']]())
    sampler.load_state_dict(torch.load(run_dir / 'sampler_state.pt'))
    _, labels, kspace = toy.validation_set(config['purity'])
    probability = toy.effective_probability(sampler(kspace, labels))[:, (1 - toy.centre_mask()).bool()]
    return float(((probability > 0.05) & (probability < 0.95)).float().mean())


def linear_probe_accuracy(features, labels, train_count=1024):
    """Held-out accuracy of a logistic regression on standardised features (first `train_count` rows train)."""
    train, test = features[:train_count], features[train_count:]
    mean, std = train.mean(0), train.std(0) + 1e-6
    train, test = (train - mean) / std, (test - mean) / std
    model = torch.nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(500):
        optimizer.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(model(train)[:, 0], labels[:train_count].float()).backward()
        optimizer.step()
    with torch.no_grad():
        return float(((model(test)[:, 0] > 0).long() == labels[train_count:]).float().mean())


def toy_feature_probe(run_dir, config):
    """Class decodability (linear probe) from a net sampler's hidden features: at its seed's initialisation and trained."""
    import toy_adaptive_sampling as toy
    image, labels = toy.images(2048, config['purity'], torch.Generator().manual_seed(31))
    kspace = toy.fft2c(image.to(torch.complex64))
    out = {}
    for name in ('initial', 'trained'):
        torch.manual_seed(config['seed'])
        net = toy.Net(net_input=config.get('net_input', 'raw'), init=config.get('net_init', 'zero'))
        if name == 'trained':
            net.load_state_dict(torch.load(run_dir / 'sampler_state.pt'))
        with torch.no_grad():
            features = net.body(net.features(kspace, labels))
        out[name] = linear_probe_accuracy(features, labels)
    return out


def toy_section(lines, summary, toy_dir):
    runs = toy_runs(toy_dir)
    if not runs:
        return
    summary['toy'] = {}
    rows = []
    for name, d in runs.items():
        f, c = d['final'], d['config']
        stochastic = toy_determinism(toy_dir / name, c)
        probe = toy_feature_probe(toy_dir / name, c) if c['sampler'] == 'net' else {}
        summary['toy'][name] = {'final': f, 'config': c, 'frac_stochastic': stochastic, 'hidden_feature_probe': probe}
        variant = c['sampler'] + (f" ({c.get('net_input', 'raw')}, head {c.get('net_init', 'zero')})" if c['sampler'] == 'net' else '')
        rows.append([name, variant, c['purity'], c['steps'], c['logit_lr'], 'frozen' if c['freeze_recon'] else 'joint',
                     fmt(f['psnr'], 2), fmt(f['class_mask_corr'], 3),
                     f"{f['own_band_share'][0]:.2f} / {f['own_band_share'][1]:.2f}", fmt(f['swap_gain_db'], 2),
                     fmt(stochastic, 3), f"{probe['initial']:.2f} / {probe['trained']:.2f}" if probe else ''])
    lines.append('\n## Toy (final evaluation of each arm)\n')
    lines.append(table(['arm', 'sampler', 'purity', 'steps', 'logit lr', 'recon', 'val PSNR', 'class-mean mask corr',
                        'own-band share A / B', 'swap cost (dB)', 'share of stochastic locations',
                        'class from hidden features (linear probe), init / trained'], rows))
    gradient_runs = sorted(toy_dir.glob('G_gradients_*/gradients.json'))
    if gradient_runs:
        rows = []
        summary['toy_gradients'] = {}
        for path in gradient_runs:
            d = json.loads(path.read_text())
            summary['toy_gradients'][path.parent.name] = d
            e, fr = d['energy'], d['fractions']
            rows.append([path.parent.name, f"{e['pop']:.2e}", f"{e['class']:.2e}", f"{e['image']:.2e}",
                         f"{d['noise_per_draw']:.2e}", fmt(fr['class'], 3), fmt(d['class_snr_per_draw'], 4),
                         fmt(d['class_offset_alignment_with_ideal'], 2)])
        lines.append('\nToy score-map gradient at the B-static mask, through the co-adapted static reconstructor or the generalist:\n')
        lines.append(table(['run', 'pop energy', 'class energy', 'image energy', 'noise per draw', 'class fraction',
                            'class SNR per draw', 'class offset vs ideal move'], rows))
    regressions = sorted(toy_dir.glob('R_*/regress.json'))
    if regressions:
        d = json.loads(regressions[0].read_text())
        summary['toy_regress'] = d
        lines.append('\nNet fitted to the table\'s per-class logit maps by direct regression:\n')
        lines.append(table(['net input', 'relative error', 'class difference, fitted / target'],
                           [[k, fmt(v['relative_error']), fmt(v['class_difference_fitted_over_target'])] for k, v in d['results'].items()]))
    probes = sorted(toy_dir.glob('probe_p*/probe.json'))
    if probes:
        rows = []
        for path in probes:
            d = json.loads(path.read_text())
            rows.append([d['purity']] + [fmt(d['accuracy'][i][m], 3) for i in ('raw', 'magnitude') for m in ('linear', 'mlp')])
        summary['toy_probe'] = [json.loads(p.read_text()) for p in probes]
        lines.append('\nClass decodability from the acquired 4x4 centre (held-out accuracy):\n')
        lines.append(table(['purity', 'raw: linear', 'raw: MLP', 'magnitude: linear', 'magnitude: MLP'], rows))
    plot_toy(runs, REPORT_DIR / 'toy.png')


TOY_CONDITIONS = (  # (run-name prefix, label): same colour in both panels
    ('D_from_static_generalist_frozen', 'frozen generalist recon, from static mask'),
    ('C_lockin_coadapted_frozen', 'frozen co-adapted recon, from static mask'),
    ('C_lockin_coadapted', 'joint, from co-adapted recon + static mask'),
    ('B_joint_scratch', 'joint from scratch'),
)


def plot_toy(runs, path):
    """Class-mean mask correlation over training (1 = one mask for both classes): table vs raw-input net, per recon condition."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (suffix, title) in zip(axes, (('table', 'per-class table (given the class)'),
                                          ('net', 'net on the raw acquired centre'))):
        for color, (prefix, label) in zip(COLORS, TOY_CONDITIONS):
            for seed, style in ((0, '-'), (1, '--')):
                name = f'{prefix}_{suffix}_s{seed}' if prefix != 'B_joint_scratch' or suffix == 'table' \
                    else f'{prefix}_net-raw_s{seed}'
                if name not in runs:
                    continue
                history = runs[name]['history']
                ax.plot([h['step'] for h in history], [h['class_mask_corr'] for h in history], style, color=color, lw=2,
                        label=label if seed == 0 else None)
        ax.set(xlabel='step', title=title)
        ax.grid(alpha=0.3, lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('corr of class-mean masks (1 = static)')
    axes[1].legend(fontsize=8, frameon=False, loc='lower left')
    fig.text(0.5, -0.02, 'solid: seed 0, dashed: seed 1', ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    lines, summary = ['# Adaptivity mechanisms, generated tables\n'], {}
    results = gradient_section(lines, summary)
    if results:
        plot_scales(results, REPORT_DIR / 'gradient_scales.png')
        radial_plot(results, REPORT_DIR / 'gradient_radial.png')
    slice_mask_section(lines, summary)
    tolerance_section(lines, summary)
    toy_section(lines, summary, args.toy_dir)
    (REPORT_DIR / 'summary.md').write_text('\n'.join(lines) + '\n')
    (REPORT_DIR / 'summary.json').write_text(json.dumps(summary, indent=1, default=str) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
