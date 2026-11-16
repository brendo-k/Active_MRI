"""Add tier F to the local v1 manifest: every usable official-train brain FLAIR volume that is not yet in it.

Tier F = S + L-only + F-only, so `datasets.train_tier=F` trains on all usable FLAIR in fastMRI multicoil_train
(339 volumes: 344 minus the 5 that fail the coil-count QC). Only FLAIR gets F-only rows; for knee and T2, tier F equals
tier L. The existing rows are kept exactly as they are: select_fastmri_subset.py no longer reproduces them (checked
2026-11-16, 144 of 1,140 rows differ), so the manifest is extended rather than regenerated. No randomness is involved.

F-only volumes live on CoreHPC only (stage 3); the rsync list is written to transfer/stage3_brain.txt.

    python scripts/data/add_flair_tier_f.py --census configs/datasets/fastmri_local_v1/census_*.json.gz
"""
import argparse
import pathlib

import pandas as pd

from select_fastmri_subset import add_sizes, load_census, qc_reasons

OUT = pathlib.Path('configs/datasets/fastmri_local_v1')
TIER = 'F-only'
STAGE = 3


def flair_tier_f_rows(census: pd.DataFrame, manifest: pd.DataFrame, min_coils: int, max_coils: int) -> pd.DataFrame:
    census = census.copy()
    census['qc'] = [qc_reasons(row, min_coils, max_coils) for row in census.itertuples()]
    rows = census[(census.cls == 'FLAIR') & (census.official_split == 'train') & (census.qc == '')
                  & ~census.file.isin(manifest.file)].copy()
    if rows.patient_id.isin(manifest.loc[manifest.split.isin(['val', 'test']), 'patient_id']).any():
        raise ValueError('an F-only patient is also in val or test')
    rows['split'] = 'train'
    rows['in_small'] = False
    rows['local'] = False
    rows = add_sizes(rows)
    rows['tier'] = TIER
    rows['stage'] = STAGE
    return rows[manifest.columns]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--census', nargs='+', required=True)
    parser.add_argument('--out', type=pathlib.Path, default=OUT)
    parser.add_argument('--min-coils', type=int, default=4)
    parser.add_argument('--max-coils', type=int, default=20)
    args = parser.parse_args()

    manifest = pd.read_csv(args.out / 'manifest.csv', keep_default_na=False)
    if (manifest.tier == TIER).any():
        raise ValueError(f'{args.out / "manifest.csv"} already has {TIER} rows')
    added = flair_tier_f_rows(load_census(args.census), manifest, args.min_coils, args.max_coils)
    combined = pd.concat([manifest, added], ignore_index=True).sort_values(['cls', 'split', 'file'], kind='stable')
    combined.to_csv(args.out / 'manifest.csv', index=False, float_format='%.4f')
    (args.out / 'transfer' / 'stage3_brain.txt').write_text(''.join(f'{p}\n' for p in sorted(added.hpc_relpath)))

    flair_train = combined[(combined.cls == 'FLAIR') & (combined.split == 'train')]
    by_field = added.groupby(['field', 'model']).size().to_frame('F-only volumes').reset_index()
    lines = ['', '## Tier F (FLAIR only, added 2026-11-16 by scripts/data/add_flair_tier_f.py)', '',
             f'Tier F = S + L-only + F-only: all {len(flair_train)} usable FLAIR volumes of official train '
             f'({len(added)} F-only, {pd.to_numeric(added.train_slices).sum()} slices, '
             f'{pd.to_numeric(added.raw_GB).sum():.0f} GB raw on CoreHPC; stage 3). Tier F FLAIR has '
             f'{pd.to_numeric(flair_train.train_slices).sum()} slices. For knee and T2, tier F is tier L.', '',
             '| ' + ' | '.join(by_field.columns) + ' |', '|' + '---|' * len(by_field.columns),
             *['| ' + ' | '.join(map(str, row)) + ' |' for row in by_field.itertuples(index=False)], '']
    with open(args.out / 'summary.md', 'a') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
