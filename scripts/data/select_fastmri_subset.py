"""Select the local fastMRI subset (knee, brain AXT2, brain AXFLAIR) from the CoreHPC header census.

Every class gets the same volume count in each tier, so any two classes (T2 + knee, FLAIR + knee, ...) pair up balanced:
  train L (large) and S (small, S inside L) from official train; val and test from official val, split by patient.
Strata: knee by protocol x field (equal cells); FLAIR by field x scanner (proportional); T2 copies FLAIR's field x
scanner counts tier by tier, so T2 vs FLAIR is not confounded by scanner. Knee train prefers the volumes already on
local scratch (batch2).

Writes manifest.csv, per-stage rsync --files-from lists and summary.md into --out.
"""
import argparse
import gzip
import json
import math
import pathlib

import numpy as np
import pandas as pd

GRID = 320
CLASSES = ('knee', 'T2', 'FLAIR')
EXCLUDE = {'file_brain_AXT2_201_2010571.h5'}  # ground truth carries ring/streak artifacts (~25 dB for every model)


def load_census(paths):
    rows = []
    for path in paths:
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rt') as f:
            census = json.load(f)
        if census['failures']:
            raise ValueError(f"{path} lists unreadable files: {census['failures'][:3]}")
        rows += census['rows']
    df = pd.DataFrame(rows)
    df['anatomy'] = np.where(df.file.str.startswith('file_brain_'), 'brain', 'knee')
    df['official_split'] = df.dir.str.extract(r'multicoil_(\w+)$', expand=False)
    df['cls'] = np.select([df.anatomy == 'knee', df.file.str.contains('_AXT2_'), df.file.str.contains('_AXFLAIR_')],
                          ['knee', 'T2', 'FLAIR'], 'other')
    df['protocol'] = np.where(df.anatomy == 'knee', np.where(df.acquisition.str.contains('PDFS'), 'PDFS', 'PD'), df.cls)
    df['field'] = np.where(df.field_T < 2.0, '1.5T', '3T')
    df['slices'] = df.kspace_shape.str[0]
    df['coils'] = df.kspace_shape.str[1]
    df['readout'] = df.kspace_shape.str[2]
    df['acq_lines'] = df.pe_max - df.pe_min + 1
    df['readout_fov_ratio'] = (df.enc_fov_mm.str[0] / df.rec_fov_mm.str[0]).round(3)
    return df[df.cls != 'other'].reset_index(drop=True)


def qc_reasons(row, min_coils, max_coils):
    reasons = []
    if row.file in EXCLUDE:
        reasons.append('artifact')
    if row.readout_fov_ratio != 2.0:
        reasons.append('readout_not_2x')
    if not min_coils <= row.coils <= max_coils:
        reasons.append('coils')
    # FLAIR is acquired below 320 lines everywhere (224 typical), so only knee and T2 must fill the grid.
    if row.cls != 'FLAIR' and (row.acq_lines < GRID or row.readout // 2 < GRID):
        reasons.append('under_grid')
    return ','.join(reasons)


def allocate(available: pd.Series, n: int, equal: bool) -> pd.Series:
    """Per-stratum counts summing to n: equal cells or proportional to `available` (largest remainder), capped by
    what each stratum has; any shortfall moves to strata with spare volumes."""
    available = available[available > 0]
    weights = pd.Series(1.0, index=available.index) if equal else available / available.sum()
    raw = weights / weights.sum() * n
    counts = np.floor(raw).astype(int)
    for key in (raw - counts).sort_values(ascending=False).index[: n - counts.sum()]:
        counts[key] += 1
    counts = np.minimum(counts, available)
    while counts.sum() < n:
        spare = available - counts
        if spare.sum() == 0:
            raise ValueError(f'only {available.sum()} volumes for {n} requested')
        counts[(spare / spare.sum()).idxmax()] += 1
    return counts


def stratified_pick(pool: pd.DataFrame, keys, counts: pd.Series, rng, prefer=None) -> pd.Index:
    picked = []
    groups = pool.groupby(keys)
    for stratum, k in counts[counts > 0].items():
        members = groups.get_group(stratum)
        members = members.iloc[rng.permutation(len(members))]
        if prefer is not None:
            members = members.sort_values(prefer, ascending=False, kind='stable')
        picked += list(members.index[:k])
    return pd.Index(picked)


def strata_keys(cls):
    return ['protocol', 'field'] if cls == 'knee' else ['field', 'model']


def pick_tier(pool, cls, n, rng, flair_counts=None, prefer=None):
    keys = strata_keys(cls)
    available = pool.groupby(keys).size()
    if cls == 'T2':
        counts = flair_counts.reindex(available.index, fill_value=0)
        if counts.sum() != n or (counts > available).any():
            raise ValueError(f'T2 cannot match the FLAIR field x scanner counts: need {flair_counts.to_dict()}')
    else:
        counts = allocate(available, n, equal=cls == 'knee')
    return stratified_pick(pool, keys, counts, rng, prefer=prefer)


def select(df, n_large, n_small, n_val, n_test, seed, local_knee):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df['local'] = df.file.isin(local_knee) & (df.anatomy == 'knee')
    df['split'] = ''
    df['in_small'] = False

    # Official val -> val/test by patient (knee patients often have both PD and PDFS).
    val_patients = df.loc[df.official_split == 'val', 'patient_id'].unique()
    test_side = set(rng.permutation(val_patients)[: len(val_patients) // 2])
    df['val_side'] = np.where(df.patient_id.isin(test_side), 'test', 'val')

    usable = df.qc == ''
    for split, side, n in (('val', 'val', n_val), ('test', 'test', n_test)):
        flair_counts = None
        for cls in ('knee', 'FLAIR', 'T2'):
            pool = df[usable & (df.cls == cls) & (df.official_split == 'val') & (df.val_side == side)]
            chosen = pick_tier(pool, cls, n, rng, flair_counts)
            df.loc[chosen, 'split'] = split
            if cls == 'FLAIR':
                flair_counts = df.loc[chosen].groupby(strata_keys('T2')).size()

    flair_large = flair_small = None
    for cls in ('knee', 'FLAIR', 'T2'):
        pool = df[usable & (df.cls == cls) & (df.official_split == 'train')]
        prefer = 'local' if cls == 'knee' else None
        large = pick_tier(pool, cls, n_large, rng, flair_large, prefer)
        df.loc[large, 'split'] = 'train'
        small = pick_tier(df.loc[large], cls, n_small, rng, flair_small, prefer)
        df.loc[small, 'in_small'] = True
        if cls == 'FLAIR':
            flair_large = df.loc[large].groupby(strata_keys('T2')).size()
            flair_small = df.loc[small].groupby(strata_keys('T2')).size()
    return df[df.split != ''].copy()


def add_sizes(sel):
    sel['train_slices'] = np.where((sel.split == 'train') & (sel.anatomy == 'knee'), np.ceil(sel.slices / 2), sel.slices).astype(int)
    sel['raw_GB'] = sel.bytes / 1e9
    sel['cache_GB'] = sel.train_slices * sel.coils * GRID * GRID * 8 / 1e9  # complex64 k-space on the 320^2 grid
    sel['maps_GB'] = sel.cache_GB / 2  # fp16 complex sensitivity maps
    sel['source'] = np.where(sel.local, 'local_batch2', 'hpc')
    sel['stage'] = np.where(sel.source == 'local_batch2', 0, np.where((sel.split != 'train') | sel.in_small, 1, 2))
    sel['tier'] = np.where(sel.split == 'train', np.where(sel.in_small, 'S', 'L-only'), sel.split)
    sel['hpc_relpath'] = 'multicoil_' + sel.official_split + '/' + sel.file
    return sel


def markdown_table(frame: pd.DataFrame) -> str:
    frame = frame.reset_index()
    header = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    rule = '|' + '---|' * len(frame.columns)
    body = ['| ' + ' | '.join(map(str, row)) + ' |' for row in frame.itertuples(index=False)]
    return '\n'.join([header, rule, *body])


def summary_markdown(sel, df, args):
    lines = [f'# fastMRI local subset v1 (seed {args.seed})', '',
             f'Per class: train L {args.n_large} (S {args.n_small} inside L), val {args.n_val}, test {args.n_test}. '
             f'Sizes: raw = h5 on CoreHPC; cache = complex64 k-space on a {GRID}x{GRID} grid (knee train every other slice, '
             'before the energy filter); maps = fp16 BART maps.', '']
    tiers = [('S', sel.split.eq('train') & sel.in_small), ('L', sel.split.eq('train')), ('val', sel.split.eq('val')),
             ('test', sel.split.eq('test'))]
    lines += ['| tier | class | volumes | slices | raw GB | cache GB | maps GB |', '|---|---|---|---|---|---|---|']
    for tier, mask in tiers:
        for cls in CLASSES:
            part = sel[mask & (sel.cls == cls)]
            lines.append(f'| {tier} | {cls} | {len(part)} | {part.train_slices.sum()} | {part.raw_GB.sum():.0f} | '
                         f'{part.cache_GB.sum():.0f} | {part.maps_GB.sum():.0f} |')
    lines += ['', '| stage | what | volumes | raw GB to copy |', '|---|---|---|---|']
    for stage, what in ((0, 'knee train already on /export/scratch (batch2)'), (1, 'S train + val + test'), (2, 'rest of L')):
        part = sel[sel.stage == stage]
        lines.append(f'| {stage} | {what} | {len(part)} | {0 if stage == 0 else part.raw_GB.sum():.0f} |')
    lines += ['', f'Cache total (L + val + test): {sel.cache_GB.sum():.0f} GB k-space + {sel.maps_GB.sum():.0f} GB maps.', '',
              '## Strata', '']
    for cls in CLASSES:
        keys = strata_keys(cls)
        table = sel[sel.cls == cls].groupby(keys + ['tier']).size().unstack(fill_value=0)
        lines += [f'### {cls}', '', markdown_table(table), '']
    lines += ['## QC exclusions (whole census)', '', df[df.qc != ''].groupby(['cls', 'qc']).size().to_frame('volumes').pipe(markdown_table), '',
              '## Coils in the selection', '', sel.groupby(['cls', 'coils']).size().unstack(fill_value=0).pipe(markdown_table), '',
              f'## FLAIR acquired phase lines and readout (the {GRID} grid is only partly filled)', '',
              sel[sel.cls == 'FLAIR'].groupby(['readout', 'acq_lines']).size().to_frame('volumes').pipe(markdown_table), '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--census', nargs='+', required=True)
    parser.add_argument('--local-knee-dir', default='/export/scratch/bkadota/fastmri_knee/batch2/multicoil_train')
    parser.add_argument('--out', required=True)
    parser.add_argument('--n-large', type=int, default=300)
    parser.add_argument('--n-small', type=int, default=80)
    parser.add_argument('--n-val', type=int, default=40)
    parser.add_argument('--n-test', type=int, default=40)
    parser.add_argument('--min-coils', type=int, default=4)
    parser.add_argument('--max-coils', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    df = load_census(args.census)
    df['qc'] = [qc_reasons(row, args.min_coils, args.max_coils) for row in df.itertuples()]
    local_knee = {p.name for p in pathlib.Path(args.local_knee_dir).glob('*.h5')}
    sel = add_sizes(select(df, args.n_large, args.n_small, args.n_val, args.n_test, args.seed, local_knee))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    columns = ['file', 'anatomy', 'cls', 'protocol', 'official_split', 'split', 'tier', 'in_small', 'stage', 'source',
               'hpc_relpath', 'patient_id', 'field', 'model', 'acquisition', 'coils', 'slices', 'train_slices', 'readout',
               'acq_lines', 'kspace_shape', 'raw_GB', 'cache_GB']
    sel.sort_values(['cls', 'split', 'file'])[columns].to_csv(out / 'manifest.csv', index=False, float_format='%.4f')
    transfer = out / 'transfer'
    transfer.mkdir(exist_ok=True)
    for stage in (1, 2):
        for anatomy in ('knee', 'brain'):
            paths = sorted(sel[(sel.stage == stage) & (sel.anatomy == anatomy)].hpc_relpath)
            (transfer / f'stage{stage}_{anatomy}.txt').write_text(''.join(f'{p}\n' for p in paths))
    (out / 'summary.md').write_text(summary_markdown(sel, df, args))
    print((out / 'summary.md').read_text())


if __name__ == '__main__':
    main()
