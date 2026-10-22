"""Build a slice-balanced dataset of knee PD, knee PDFS and brain T2 as symlinks.

fastMRI knee has no T2, so T2 comes from brain AXT2. Each class gets (nearly) the
same number of raw slices per split; volumes are random (seeded) and whole, so no
scan is shared between train and val. Layout:

    <out>/{train,val}/{knee-PD,knee-PDFS,brain-T2}/<file>.h5 -> original
    <out>/manifest.json
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np

CLASSES = {  # class -> (anatomy dir, acquisition prefix match)
    'knee-PD': ('knee', lambda a: a.startswith('CORPD') and 'FS' not in a),
    'knee-PDFS': ('knee', lambda a: a.startswith('CORPDFS')),
    'brain-T2': ('brain', lambda a: a == 'AXT2'),
}
SPLITS = {'train': 'multicoil_train', 'val': 'multicoil_val'}


def scan(root, anatomy, split_dir, match):
    out = []
    for f in sorted(glob.glob(f'{root}/{anatomy}/data/{split_dir}/*.h5')):
        with h5py.File(f, 'r') as h:
            if match(str(h.attrs.get('acquisition', ''))):
                out.append((f, int(h['kspace'].shape[0])))
    return out


def pick(vols, target, seed):
    """Random whole volumes whose slice total is as close to `target` as possible without exceeding it."""
    order = np.random.RandomState(seed).permutation(len(vols))
    chosen, total = [], 0
    for i in order:
        if total + vols[i][1] <= target:
            chosen.append(vols[i])
            total += vols[i][1]
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=os.path.expanduser('~/scratch/data/fastmri'))
    p.add_argument('--out', default=os.path.expanduser('~/scratch/data/fastmri/balanced_pd_pdfs_t2'))
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()

    manifest = {}
    for split, split_dir in SPLITS.items():
        pools = {c: scan(args.root, a, split_dir, m) for c, (a, m) in CLASSES.items()}
        target = min(sum(n for _, n in v) for v in pools.values())
        manifest[split] = {}
        for c, vols in pools.items():
            chosen = pick(vols, target, args.seed)
            d = os.path.join(args.out, split, c)
            os.makedirs(d, exist_ok=True)
            for f, _ in chosen:
                link = os.path.join(d, os.path.basename(f))
                if not os.path.lexists(link):
                    os.symlink(f, link)
            manifest[split][c] = {'volumes': len(chosen), 'slices': sum(n for _, n in chosen),
                                  'available_volumes': len(vols), 'files': [os.path.basename(f) for f, _ in chosen]}
        print(split, 'target', target, {c: (m['volumes'], m['slices']) for c, m in manifest[split].items()})
    with open(os.path.join(args.out, 'manifest.json'), 'w') as fh:
        json.dump({'seed': args.seed, **manifest}, fh, indent=1)


if __name__ == '__main__':
    main()
