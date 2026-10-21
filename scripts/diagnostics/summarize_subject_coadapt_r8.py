"""Tables for subject_coadapt_ceiling_r8.py and subject_ssl_tta_r8.py: per-arm PSNR change vs base, subject-specific
gain (arm minus its other-subject control), paired per slice, 95% CIs bootstrapping volumes. "Typical" drops slices
whose target signal lies more than 30% outside the sense support (the mask-independent tail).

    python scripts/diagnostics/summarize_subject_coadapt_r8.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path('analysis/subject_coadapt')
BOOTSTRAPS = 2000
TAIL_UNSUPPORTED = 0.3
FAMILIES = ('param', 'grid')
PARAM_NAMES = ('scale', 'anisotropy_a', 'anisotropy_b', 'tilt')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ceiling-dirs', default='brain,knee', help='folders under analysis/subject_coadapt, one table each')
    parser.add_argument('--ssl-dirs', default='', help='comma-separated folders under ssl/, e.g. brain,knee')
    parser.add_argument('--out', type=Path, default=ROOT / 'summary.md')
    parser.add_argument('--exclude', default='', help='comma-separated volume file names left out of the ceiling tables')
    return parser.parse_args()


def load(folder, exclude=()):
    volumes = [json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))]
    return [v for v in volumes if v.get('volume') not in exclude]


def bootstrap(per_volume, rng):
    per_volume = np.asarray(per_volume)
    draws = [per_volume[rng.randint(len(per_volume), size=len(per_volume))].mean() for _ in range(BOOTSTRAPS)]
    return np.percentile(draws, [2.5, 97.5])


def contrast(volumes, arm, reference, keep, rng):
    """Slice-weighted mean of arm - reference over kept slices, and a volume-bootstrap CI of the volume means."""
    deltas, per_volume = [], []
    for v in volumes:
        if arm not in v['psnr'] or reference not in v['psnr']:
            continue
        a, r = np.asarray(v['psnr'][arm]), np.asarray(v['psnr'][reference][:len(v['psnr'][arm])])
        k = keep(v)[:len(a)]
        if k.sum() == 0:
            continue
        deltas.extend((a - r)[k])
        per_volume.append((a - r)[k].mean())
    if not per_volume:
        return None
    low, high = bootstrap(per_volume, rng)
    return f'{np.mean(deltas):+.3f} [{low:+.3f}, {high:+.3f}]'


def keepers(volumes):
    every = lambda v: np.ones(len(v['psnr']['base']), dtype=bool)  # noqa: E731
    typical = lambda v: np.asarray(v.get('unsupported', [0] * len(v['psnr']['base']))) <= TAIL_UNSUPPORTED  # noqa: E731
    return {'all': every, 'typical': typical}


def table(title, volumes_by_anatomy, rows, rng):
    lines = [f'### {title}', '', '| contrast | ' + ' | '.join(
        f'{anatomy} {subset}' for anatomy in volumes_by_anatomy for subset in ('all', 'typical')) + ' |',
        '|---|' + '---|' * (2 * len(volumes_by_anatomy))]
    for label, arm, reference in rows:
        cells = []
        for volumes in volumes_by_anatomy.values():
            for keep in keepers(volumes).values():
                cells.append(contrast(volumes, arm, reference, keep, rng) or '')
        if any(cells):
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    counts = ', '.join(f'{a}: {len(v)} volumes, {sum(len(x["psnr"]["base"]) for x in v)} slices'
                       for a, v in volumes_by_anatomy.items())
    return lines + ['', counts, '']


def ceiling_rows(steps):
    rows = [('mask (fit on F) - base', 'mask', 'base'),
            ('mask_in (per-volume mask oracle, fit on F+H) - base', 'mask_in', 'base'),
            ('mask_ctrl (other subjects) - base', 'mask_ctrl', 'base'),
            ('mask - mask_ctrl (subject-specific)', 'mask', 'mask_ctrl')]
    for family in FAMILIES:
        rows += [(f'{family} (fit on F) - base', family, 'base'),
                 (f'{family}_in (fit on F+H) - base', f'{family}_in', 'base'),
                 (f'{family}_ctrl (other subjects) - base', f'{family}_ctrl', 'base'),
                 (f'{family} - {family}_ctrl (subject-specific)', family, f'{family}_ctrl'),
                 (f'{family}_in - mask_in (family vs free per-volume oracle)', f'{family}_in', 'mask_in')]
    for step in steps:
        rows += [(f'recon@{step} - base', f'recon@{step}', 'base'),
                 (f'recon_generic@{step} - base (continued training)', f'recon_generic@{step}', 'base'),
                 (f'recon@{step} - recon_generic@{step} (subject vs continued training)', f'recon@{step}', f'recon_generic@{step}'),
                 (f'recon_ctrl@{step} - base', f'recon_ctrl@{step}', 'base'),
                 (f'recon@{step} - recon_ctrl@{step} (subject-specific)', f'recon@{step}', f'recon_ctrl@{step}'),
                 (f'joint@{step} - base', f'joint@{step}', 'base'),
                 (f'joint@{step} - joint_ctrl@{step} (subject-specific)', f'joint@{step}', f'joint_ctrl@{step}'),
                 (f'joint@{step} - recon@{step} (mask gain on a co-adapted VarNet)', f'joint@{step}', f'recon@{step}'),
                 (f'recon_on_mask_in@{step} - base', f'recon_on_mask_in@{step}', 'base'),
                 (f'recon_on_mask_in@{step} - mask_in (VarNet gain on the oracle mask)', f'recon_on_mask_in@{step}', 'mask_in')]
    return rows


def pearson(x, y):
    x, y = np.asarray(x, float) - np.mean(x), np.asarray(y, float) - np.mean(y)
    return float((x * y).sum() / max(np.sqrt((x ** 2).sum() * (y ** 2).sum()), 1e-30))


def theta_reliability(folder, volumes):
    """Is a volume's fitted family parameter a property of the volume? The F fit against the H fit of the same volume,
    across volumes: per parameter for `param`, and for the `grid` map the within-volume correlation of the two fits'
    anatomy-centred grids against the between-volume one (null). `ctrl` (other subjects) against H is a second null."""
    volumes = [v for v in volumes if 'theta' in v and all(a in v['theta'] for a in ('param', 'param_h'))]
    if len(volumes) < 3:
        return []
    lines = [f'### Per-volume family parameters, {folder}: reliability across {len(volumes)} volumes', '',
             '| parameter | mean F fit | sd across volumes (F fit) | corr(F fit, H fit) | corr(ctrl fit, H fit) |',
             '|---|---|---|---|---|']
    for i, name in enumerate(PARAM_NAMES):
        f, h = [v['theta']['param'][i] for v in volumes], [v['theta']['param_h'][i] for v in volumes]
        ctrl = [v['theta'].get('param_ctrl', [np.nan] * 4)[i] for v in volumes]
        lines.append(f'| {name} | {np.mean(f):+.3f} | {np.std(f):.3f} | {pearson(f, h):+.2f} | {pearson(ctrl, h):+.2f} |')
    grids = [v for v in volumes if all(a in v['theta'] for a in ('grid', 'grid_h'))]
    if len(grids) >= 3:
        f = np.array([v['theta']['grid'] for v in grids])
        h = np.array([v['theta']['grid_h'] for v in grids])
        f, h = f - f.mean(0), h - h.mean(0)
        within = np.mean([pearson(f[i], h[i]) for i in range(len(grids))])
        between = np.mean([pearson(f[i], h[j]) for i in range(len(grids)) for j in range(len(grids)) if i != j])
        lines += ['', f'grid offset (centred on the mean over volumes): corr(F fit, H fit) within a volume {within:+.2f}, '
                      f'between volumes {between:+.2f} ({len(grids)} volumes)']
    return lines + ['']


def ssl_rows(volumes):
    keys = sorted({k for v in volumes for k in v['psnr'] if k != 'base'}, key=lambda k: (k.split('@')[0], k.split('@')[1]))
    return [(f'{k} - base', k, 'base') for k in keys] + \
        [(f'ssl_vol{s} - ssl_ctrl{s} (subject-specific)', f'ssl_vol{s}', f'ssl_ctrl{s}')
         for s in sorted({k[len('ssl_vol'):] for v in volumes for k in v['psnr'] if k.startswith('ssl_vol')})]


def ssl_overview(folders):
    """One row per saved self-supervised configuration: mean PSNR change vs base of the per-slice fit at each step."""
    rows, steps = [], set()
    for folder in folders:
        volumes = load(folder)
        if not volumes:
            continue
        config = volumes[0]['config']
        method = config.get('method', 'ssdu')
        loss = config.get('echo_loss') if method == 'echo' else config.get('loss_space', 'image')
        deltas = {}
        for key in sorted({k for v in volumes for k in v['psnr'] if k.startswith('ssl_slice@') and k != 'ssl_slice@stop'}):
            per_slice = [a - b for v in volumes for a, b in zip(v['psnr'].get(key, []), v['psnr']['base'])]
            if per_slice:
                deltas[int(key.split('@')[1])] = np.mean(per_slice)
        steps |= set(deltas)
        sens = 'frozen' if config.get('freeze_sens') else 'trained'
        rows.append((method, loss, config['lr'], config['rho'] if method == 'ssdu' or loss == 'heldout' else '', sens,
                     sum(len(v['psnr']['base'][:config.get('slice_count', 4)]) for v in volumes), deltas))
    steps = sorted(steps)
    lines = ['### Self-supervised per-slice adaptation (PSNR change vs base, dB, mean over slices)', '',
             '| method | loss | lr | rho | sense net | slices | ' + ' | '.join(f'@{s}' for s in steps) + ' |',
             '|---|---|---|---|---|---|' + '---|' * len(steps)]
    for method, loss, lr, rho, sens, n, deltas in sorted(rows, key=lambda r: (r[0], str(r[1]), r[2])):
        lines.append(f'| {method} | {loss} | {lr:g} | {rho} | {sens} | {n} | ' +
                     ' | '.join(f'{deltas[s]:+.3f}' if s in deltas else '' for s in steps) + ' |')
    return lines + ['']


def oracle_saturation(anatomy, steps_folders, reference):
    """Per-volume mask oracle (`mask_in`) and its out-of-sample twin (`mask`, fitted on F only) by fit length, over the
    volumes of the longest fit only, so every row compares the same slices."""
    names = [p.name for p in sorted((ROOT / reference).glob('*.json'))]
    lines = [f'### Per-volume mask oracle vs fit length, {anatomy} (same volumes as {reference})', '',
             '| fit steps | folder | arm | volumes | PSNR vs base (dB) |', '|---|---|---|---|---|']
    for steps, folder in steps_folders:
        for arm in ('mask_in', 'mask'):
            deltas = [np.array(v['psnr'][arm]) - np.array(v['psnr']['base'])
                      for v in (json.loads((ROOT / folder / n).read_text()) for n in names if (ROOT / folder / n).exists())
                      if arm in v['psnr']]
            if deltas:
                lines.append(f'| {steps} | {folder} | {arm} | {len(deltas)} of {len(names)} | {np.concatenate(deltas).mean():+.3f} |')
    return lines + ['']


def main():
    args = parse_args()
    rng = np.random.RandomState(0)
    out = ['# Per-subject co-adaptation (R=8, LOUPE VarNet uo8d68n0)', '',
           'Paired per-slice PSNR differences (dB), slice-weighted mean [95% CI over volumes]. Scored on the odd',
           'half H of each validation volume; F = the even half. Typical = signal outside the sense support <= 30%.', '']
    exclude = set(filter(None, args.exclude.split(',')))
    if exclude:
        out += [f'Left out of the ceiling tables: {", ".join(sorted(exclude))}', '']
    ceiling = {d: load(ROOT / d, exclude) for d in args.ceiling_dirs.split(',') if (ROOT / d).is_dir()}
    ceiling = {d: v for d, v in ceiling.items() if v}
    for d, volumes in ceiling.items():
        steps = sorted({int(k.split('@')[1]) for v in volumes for k in v['psnr'] if '@' in k})
        distances = [x for v in volumes for x in v.get('distance_to_fit', [])]
        title = f'Ground-truth ceiling, {d} (fit on F, score H' + (f'; slice distance to F {min(distances)}-{max(distances)})' if distances else ')')
        out += table(title, {d: volumes}, ceiling_rows(steps), rng)
        out += theta_reliability(d, volumes)
    if ceiling:
        corr = {d: {arm: np.mean([v['mask_corr_with_base'][arm] for v in vs if arm in v.get('mask_corr_with_base', {})])
                    for arm in ('mask', 'mask_in', 'mask_ctrl', 'joint', 'joint_ctrl')} for d, vs in ceiling.items()}
        out += ['Mean correlation of the fitted per-volume probability map with the LOUPE map: ' +
                '; '.join(f'{d}: ' + ', '.join(f'{a} {c:.4f}' for a, c in cs.items() if np.isfinite(c)) for d, cs in corr.items()), '']
    for anatomy in ('brain', 'knee'):
        folders = [(150, f'{anatomy}_block_full'), (600, f'{anatomy}_block_maskin600'), (1500, f'{anatomy}_block_maskin1500'),
                   (1500, f'{anatomy}_block_mask1500')]
        folders = [(steps, f) for steps, f in folders if (ROOT / f).is_dir() and any((ROOT / f).glob('*.json'))]
        if any(f.endswith('maskin1500') for _, f in folders):
            out += oracle_saturation(anatomy, folders, f'{anatomy}_block_maskin1500')
    ssl_folders = sorted(p for p in (ROOT / 'ssl').glob('*') if p.is_dir())
    if ssl_folders:
        out += ssl_overview(ssl_folders)
    for d in filter(None, args.ssl_dirs.split(',')):
        volumes = load(ROOT / 'ssl' / d)
        if volumes:
            stops = [s['stop_step'] for v in volumes for s in ([v['stop']['ssl_vol']] if 'ssl_vol' in v['stop'] else [])]
            out += table(f'ZS-SSL, {d}', {d: volumes}, ssl_rows(volumes), rng)
            out += [f'ssl_vol Gamma-selected stop steps: {stops}', '']
    args.out.write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
