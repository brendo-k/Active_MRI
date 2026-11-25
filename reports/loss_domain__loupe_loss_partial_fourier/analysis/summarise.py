"""Tables and figures of loss_domain__loupe_loss_partial_fourier from its score files, layout and probe results and
training logs.

    tables   ../summary.json and ../summary.txt: guards, pooled metrics, layout, value of the half, probe, criteria. No GPU.
    curves   ../curves.json and ../figures/curves.png: the trainer's per-epoch validation PSNR and training loss. No GPU.
    maps     ../figures/learned_maps.png: each run's learned map p(k) = sigmoid(slope * logits). Needs the checkpoints.
    extent   analysis/extent.json: per side of DC, the share of the layout window holding acquired k-space (any coil
             non-zero) on every dev slice and every local tier-S training slice. Needs the data, no GPU.
    recons   ../figures/recons.png: the median-PSNR dev slice, each seed-42 run's mask, reconstruction and error on one
             scale. Needs the checkpoints, the data and a GPU. Loads train tier S only to build the loaders (tier F's
             train files are not on the lab server); the dev set does not depend on it.

    python reports/loss_domain__loupe_loss_partial_fourier/analysis/summarise.py tables curves
    python reports/loss_domain__loupe_loss_partial_fourier/analysis/summarise.py maps extent recons \
        --runs-dir /export/scratch/bkadota/loss_domain__loupe_loss_partial_fourier_sync/runs \
        --data-root 'hpc=/export/scratch/bkadota/fastmri_raw/{anatomy}'
"""
import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / 'scripts'), str(REPO / 'scripts' / 'eval'), str(REPO / 'scripts' / 'diagnostics')]

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPORT = Path(__file__).resolve().parents[1]
FIGURES = REPORT / 'figures'
ARMS = ('Lrss', 'Lcplx', 'Lrsssym', 'Lcplxsym')
SEEDS = (42, 43)
LABELS = {
    'Lrss': 'RSS loss, free map (control)',
    'Lcplx': 'complex loss, free map',
    'Lrsssym': 'RSS loss, point-symmetric map',
    'Lcplxsym': 'complex loss, point-symmetric map',
}
COLOURS = {'Lrss': '#2563c9', 'Lcplx': '#d4761f', 'Lrsssym': '#7fa6e6', 'Lcplxsym': '#efb27a'}
LOGS = {  # jobs.txt, run state 2
    ('Lrss', 42): 3161844, ('Lcplx', 42): 3161845, ('Lrsssym', 42): 3161846, ('Lcplxsym', 42): 3161847,
    ('Lrss', 43): 3161848, ('Lcplx', 43): 3161849, ('Lrsssym', 43): 3161850, ('Lcplxsym', 43): 3161851,
}
HALF_PAIRS = {'rss': ('Lrss', 'Lrsssym'), 'cplx': ('Lcplx', 'Lcplxsym')}
PROBE_FSCR1 = ('L_1_s42_fscr1_tF', 'F_4_s42_fscr1_tF', 'Fb_4_s42_fscr1_tF')


def run_name(arm, seed):
    return f'{arm}_1_s{seed}_tF'


def load_json(name):
    return json.loads((REPORT / name).read_text())


def probe_rows():
    rows = {}
    for path in [REPORT / f'analysis/probe_{arm}.json' for arm in ARMS] + [REPORT / 'probe_fscr1.json']:
        for name, result in load_json(path.relative_to(REPORT))['runs'].items():
            summary = result['summary']
            rows[name] = {
                'partner_anti': summary['partner']['anti_fraction_median'],
                'partner_response': summary['partner']['response_median'],
                'placebo_rows_anti': summary['placebo_rows']['anti_fraction_median'],
                'placebo_columns_anti': summary['placebo_columns']['anti_fraction_median'],
                'margins': {point: summary['partner']['anti_fraction_median'] - summary[point]['anti_fraction_median']
                            for point in ('placebo_rows', 'placebo_columns')},
            }
    return rows


def tables():
    scores = {(arm, seed): load_json(f'scores/{run_name(arm, seed)}.json') for arm in ARMS for seed in SEEDS}
    layout = load_json('analysis/layout.json')
    contrasts = load_json('contrasts.json')['contrasts']
    probe = probe_rows()
    pooled = {f'{arm}_s{seed}': {key: value['all'] for key, value in scores[arm, seed]['means'].items()}
              for arm in ARMS for seed in SEEDS}
    guards = {}
    for seed in SEEDS:
        ssim = {arm: pooled[f'{arm}_s{seed}']['ssim'] for arm in ARMS}
        guards[f's{seed}'] = {
            'ssim_at_least_0.80': all(value >= 0.80 for value in ssim.values()),
            'rsssym_within_0.03_of_rss': abs(ssim['Lrsssym'] - ssim['Lrss']) <= 0.03,
            'cplxsym_within_0.03_of_cplx': abs(ssim['Lcplxsym'] - ssim['Lcplx']) <= 0.03,
            'cplx_within_0.03_of_rss': abs(ssim['Lcplx'] - ssim['Lrss']) <= 0.03,
            'realised_fraction_within_0.002': all(abs(pooled[f'{arm}_s{seed}']['realised_fraction'] - 0.125) <= 0.002
                                                  for arm in ARMS),
            'reconstructions_below_20dB': {arm: len(scores[arm, seed]['guards']['failed_reconstructions_below_20dB'])
                                           for arm in ARMS},
        }
    right = {f'{arm}_s{seed}': layout[run_name(arm, seed)]['right_share_primary'] for arm in ARMS for seed in SEEDS}
    half = {f'{loss}_s{seed}': contrasts[f'half_{loss}_s{seed}']['psnr']['all'] for loss in HALF_PAIRS for seed in SEEDS}
    rss_right = [right[f'Lrss_s{seed}'] for seed in SEEDS]
    cplx_right = [right[f'Lcplx_s{seed}'] for seed in SEEDS]
    layout_verdict = ('the loss decides the layout' if max(rss_right) <= 0.15 and min(cplx_right) >= 0.30
                      else 'the loss does not decide the layout' if max(cplx_right) <= 0.15 else 'inconclusive')
    rss_half = [half[f'rss_s{seed}'] for seed in SEEDS]
    half_pass = all(row['mean'] >= 0.10 and row['ci95'][0] > 0 for row in rss_half)
    mechanism_pass = all(
        probe[run_name('Lrss', seed)]['partner_anti'] >= 0.90 and probe[run_name('Lrss', seed)]['partner_response'] >= 0.10
        and min(probe[run_name('Lrss', seed)]['margins'].values()) >= 0.30 for seed in SEEDS)
    criteria = {
        'layout': {'rule': 'Lrss <= 0.15 and Lcplx >= 0.30 in both seeds: loss decides; Lcplx <= 0.15 in both: it does not',
                   'Lrss': rss_right, 'Lcplx': cplx_right, 'verdict': layout_verdict},
        'value_of_half_rss': {'rule': 'Lrss - Lrsssym >= +0.10 dB, CI above 0, both seeds', 'values': rss_half,
                              'met': half_pass},
        'value_of_half_cplx': {'rule': 'reported; expected to include 0', 'values': [half[f'cplx_s{seed}'] for seed in SEEDS]},
        'mechanism': {'rule': 'Lrss both seeds: -k antilinear >= 0.90, response >= 0.10, >= 0.30 above each placebo',
                      'met': mechanism_pass},
    }
    summary = {'pooled': pooled, 'guards': guards, 'right_share': right, 'value_of_half': half, 'probe': probe,
               'criteria': criteria}
    (REPORT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['run                  PSNR dB  SSIM    NMSE     fraction  right-of-DC share']
    for arm in ARMS:
        for seed in SEEDS:
            row = pooled[f'{arm}_s{seed}']
            lines.append(f"{run_name(arm, seed):20s} {row['psnr']:7.3f}  {row['ssim']:.4f}  {row['nmse']:.5f}  "
                         f"{row['realised_fraction']:.4f}    {right[f'{arm}_s{seed}']:.3f}")
    lines.append('')
    for key, row in half.items():
        lines.append(f"value of the half {key}: {row['mean']:+.3f} dB [{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]")
    lines.append('')
    for name, row in probe.items():
        margins = ', '.join(f'{key} {value:.3f}' for key, value in row['margins'].items())
        lines.append(f"probe {name:20s} -k antilinear {row['partner_anti']:.3f} response {row['partner_response']:.3f}; "
                     f"-k minus placebo: {margins}")
    lines.append('')
    lines.append(f"layout: {layout_verdict}; value of the half (RSS) met: {half_pass}; mechanism met: {mechanism_pass}")
    (REPORT / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def parse_log(path):
    """Per epoch: the trainer's validation PSNR and SSIM (the [brain] line before each epoch line) and both losses."""
    epochs, metrics = [], None
    for line in path.read_text().splitlines():
        found = re.search(r'\[brain\] .*PSNR=([\d.]+) SSIM=([\d.]+)', line)
        if found:
            metrics = (float(found.group(1)), float(found.group(2)))
        found = re.search(r'Epoch = \[\s*(\d+)/\s*\d+\] TrainLoss = ([\d.eE+-]+) ValLoss = ([\d.eE+-]+)', line)
        if found and metrics:
            epochs.append({'epoch': int(found.group(1)), 'val_psnr': metrics[0], 'val_ssim': metrics[1],
                           'train_loss': float(found.group(2)), 'val_loss': float(found.group(3))})
            metrics = None
    return epochs


def curves():
    data = {run_name(arm, seed): parse_log(REPORT / f'logs/loupe_lpf-{job}.log') for (arm, seed), job in LOGS.items()}
    (REPORT / 'curves.json').write_text(json.dumps(data) + '\n')
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for column, losses in enumerate((('Lrss', 'Lrsssym'), ('Lcplx', 'Lcplxsym'))):
        for arm in losses:
            for seed, dash in zip(SEEDS, ('-', '--')):
                rows = data[run_name(arm, seed)]
                epochs = [row['epoch'] for row in rows]
                axes[0, column].plot(epochs, [row['val_psnr'] for row in rows], dash, color=COLOURS[arm], lw=1.5,
                                     label=f'{arm} s{seed}')
                axes[1, column].plot(epochs, [row['train_loss'] for row in rows], dash, color=COLOURS[arm], lw=1.5)
        axes[0, column].set_title('RSS loss' if column == 0 else 'complex loss (val PSNR on the trainer\'s own target)')
        axes[0, column].legend(fontsize=8, loc='lower right', ncol=2)
        axes[1, column].set_xlabel('epoch')
        for axis in axes[:, column]:
            axis.axvline(90, color='#999999', lw=0.8)
            axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel('trainer val PSNR (dB, 307 dev slices)')
    axes[1, 0].set_ylabel('training loss (L1, each panel its own loss)')
    figure.suptitle('Training curves (solid seed 42, dashed seed 43; grey line: lr x0.1 at epoch 90)')
    figure.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES / 'curves.png', dpi=110)
    for name, rows in data.items():
        print(f"{name:20s} epochs {rows[0]['epoch']}-{rows[-1]['epoch']}, last val PSNR {rows[-1]['val_psnr']:.2f} dB")


def maps(args):
    from loupe_map_asymmetry import learned_map

    figure, axes = plt.subplots(2, 4, figsize=(17, 9))
    for column, arm in enumerate(ARMS):
        for row, seed in enumerate(SEEDS):
            p = learned_map(args.runs_dir / run_name(arm, seed))
            image = axes[row, column].imshow(p, cmap='magma', vmin=0, vmax=1)
            for offset in (-98, -32, 32, 98):
                axes[row, column].axvline(160 + offset, color='#4fd1c5', lw=0.6, ls=':')
            axes[row, column].axhline(33, color='#4fd1c5', lw=0.6, ls=':')
            axes[row, column].axhline(287, color='#4fd1c5', lw=0.6, ls=':')
            axes[row, column].set_title(f'{run_name(arm, seed)}\n{LABELS[arm]}', fontsize=9)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.colorbar(image, ax=axes, fraction=0.015, label='p(k) = sigmoid(slope * logits), before budget rescaling')
    figure.suptitle('Learned LOUPE maps (columns = phase encode; dotted: the layout window, 32-98 columns from DC, '
                    'rows 33-287)')
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES / 'learned_maps.png', dpi=100)


def window_acquired_by_side(kspace):
    """[N, ...coil k-space] -> [N, 2] share of the layout window acquired left and right of DC (any coil non-zero)."""
    import torch

    magnitude = kspace.abs() if kspace.is_complex() else kspace.abs().sum(dim=-1)
    acquired = (magnitude.flatten(1, -3).sum(dim=1) > 0) if magnitude.dim() > 3 else magnitude > 0
    rows = slice(160 - 127, 160 + 128)
    distance = torch.arange(acquired.shape[-1], device=acquired.device) - 160
    inside = (distance.abs() >= 32) & (distance.abs() <= 98)
    window = acquired[:, rows]
    left = window[..., inside & (distance < 0)].float().mean(dim=(1, 2))
    right = window[..., inside & (distance > 0)].float().mean(dim=(1, 2))
    return torch.stack([left, right], dim=1)


def extent(args):
    import torch

    import eval_multistep_adaptivity as ema

    data_args = SimpleNamespace(workers=4, val_half='dev', data_root=args.data_root, chunk=8)
    cfg = ema.datasets_config(args.runs_dir / run_name('Lrss', 42), data_args)
    cfg.datasets.train_tier = 'S'
    train_loader, val_loader = ema.build_dataloaders(cfg)
    result = {}
    for split, dataset, collate in (('dev', val_loader.dataset, val_loader.collate_fn),
                                    ('train_tier_S', train_loader.dataset, train_loader.collate_fn)):
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate)
        shares = torch.cat([window_acquired_by_side(batch['kspace']) for batch in loader]).numpy()
        result[split] = {
            'slices': len(shares),
            'mean_share_acquired': {'left': float(shares[:, 0].mean()), 'right': float(shares[:, 1].mean())},
            'min_share_acquired': {'left': float(shares[:, 0].min()), 'right': float(shares[:, 1].min())},
            'slices_fully_acquired': {'left': int((shares[:, 0] == 1).sum()), 'right': int((shares[:, 1] == 1).sum())},
        }
        print(split, json.dumps(result[split]))
    (REPORT / 'analysis' / 'extent.json').write_text(json.dumps(result, indent=2) + '\n')


def median_slice():
    npz = np.load(REPORT / f"scores/{run_name('Lrss', 42)}.npz", allow_pickle=True)
    psnr = npz['psnr'][:, 0]
    index = int(np.argsort(psnr)[len(psnr) // 2])
    return Path(str(npz['volume'][index])).name, int(npz['slice_id'][index])


def recons(args):
    import torch

    import eval_multistep_adaptivity as ema
    from activemri.utils.multistep import masked_slice_metrics, step_thresholds

    volume, slice_id = median_slice()
    data_args = SimpleNamespace(workers=0, val_half='dev', data_root=args.data_root, chunk=1)
    outputs = {}
    with torch.no_grad():
        for arm in ARMS:
            run = args.runs_dir / run_name(arm, 42)
            model, checkpoint = ema.load_run(run, 'last_model.pt', 'cuda')
            options = checkpoint['options']
            cfg = ema.datasets_config(run, data_args)
            cfg.datasets.train_tier = 'S'
            _, val_loader = ema.build_dataloaders(cfg)
            dataset, collate = val_loader.dataset, val_loader.collate_fn
            parts = dataset.datasets if hasattr(dataset, 'datasets') else [dataset]
            examples = [example for part in parts for example in part.examples]
            index = next(i for i, (fname, sid) in enumerate(examples) if Path(str(fname)).name == volume and sid == slice_id)
            part = ema.labelled(collate([dataset[index]]), options, 'cuda')
            sense, _ = model.sense_maps_for(part['kspace'])
            masks, _ = model.sample_trajectory(part['kspace'], part['labels'],
                                               step_thresholds(1234, [index], model.num_step, model.shape, 'cuda'))
            output = model.reconstruct_from_mask(part['kspace'], masks[-1], sense, part['labels'])
            scores = masked_slice_metrics(output, part['target'], part['metric_mask'])
            outputs[arm] = {'mask': masks[-1][0, 0].cpu().numpy(), 'recon': output[0, 0].float().cpu().numpy(),
                            'target': part['target'][0, 0].float().cpu().numpy(),
                            'metric_mask': part['metric_mask'][0, 0].cpu().numpy(), 'psnr': scores['psnr'].item()}
            del model
            torch.cuda.empty_cache()
    target = outputs['Lrss']['target']
    scale = float(np.percentile(target, 99.5))
    error_scale = 0.2 * scale
    figure, axes = plt.subplots(3, 5, figsize=(19, 11.5))
    axes[0, 0].imshow(target, cmap='gray', vmin=0, vmax=scale)
    axes[0, 0].set_title('fully sampled target (RSS)')
    for row in (1, 2):
        axes[row, 0].axis('off')
    for column, arm in enumerate(ARMS, start=1):
        result = outputs[arm]
        axes[0, column].imshow(result['mask'], cmap='gray')
        axes[0, column].set_title(f'{run_name(arm, 42)}\n{LABELS[arm]}\nmask, draw seed 1234', fontsize=9)
        axes[1, column].imshow(result['recon'], cmap='gray', vmin=0, vmax=scale)
        axes[1, column].set_title(f"reconstruction, PSNR {result['psnr']:.2f} dB (metric mask)", fontsize=9)
        error = np.abs(result['recon'] - result['target']) * (result['metric_mask'] > 0)
        image = axes[2, column].imshow(error, cmap='inferno', vmin=0, vmax=error_scale)
        axes[2, column].set_title('|recon - target|, metric mask', fontsize=9)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axes[2, 1:], fraction=0.02, label='error, same scale for every run (0-20% of target p99.5)')
    figure.suptitle(f'{volume} slice {slice_id} (median-PSNR dev slice of Lrss_1_s42_tF); RSS-magnitude scoring for '
                    'every arm')
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES / 'recons.png', dpi=100)
    print({arm: round(result['psnr'], 3) for arm, result in outputs.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stages', nargs='+', choices=('tables', 'curves', 'maps', 'extent', 'recons'))
    parser.add_argument('--runs-dir', type=Path)
    parser.add_argument('--data-root', action='append', default=[], metavar='SOURCE=DIR')
    args = parser.parse_args()
    stages = {'tables': lambda: tables(), 'curves': lambda: curves(), 'maps': lambda: maps(args), 'extent': lambda: extent(args),
              'recons': lambda: recons(args)}
    if {'maps', 'extent', 'recons'} & set(args.stages) and args.runs_dir is None:
        parser.error('maps, extent and recons need --runs-dir')
    for stage in args.stages:
        stages[stage]()


if __name__ == '__main__':
    main()
