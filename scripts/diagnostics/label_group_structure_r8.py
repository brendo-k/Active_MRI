"""Is there protocol- or volume-level structure in the per-slice oracle labels of oracle_labels_r8.py?

Two groupings finer than anatomy, each a mask granularity a scanner can deploy:

  protocol  brain / knee-PD / knee-PDFS, known from the scan header: one mask per protocol
  volume    one mask per volume, the only adaptive granularity a 3D (SPACE) scan can use, since every readout
            position shares the phase-encode mask

Labels are compared as sampling probabilities on the locations every label can sample (not preselected, inside the
valid region). Blurring is in density space (normalised convolution of the probability), so it keeps each region's
sampling density, unlike a blur of logits.

  protocol reliability   knee volumes are split at random into two halves, stratified by protocol; each half gives a
                         PD-minus-PDFS mean label; reliability is the correlation of the two halves' differences,
                         averaged over splits. The null permutes protocol across knee volumes (same counts).
                         Brain-minus-knee is the reference for a difference known to matter.
  difference energy      squared size of a group difference without sampling noise: the mean, over splits, of the
                         dot product of the two halves' differences (each half estimates it independently).
                         PD-minus-PDFS is read against brain-minus-knee, whose mask value is measured.
  explained share        share of the labels' variance (knee around the knee mean; all around the global mean) that
                         the group means explain, against the same share under permuted groups (finite-sample bias).
  volume reliability     each volume's labels are split into two halves (alternating slices, or first and second
                         block of slices); each half's mean deviation from its protocol mean (leaving the volume out)
                         is correlated with the other half's (within). The null pairs halves of two different volumes
                         of the same protocol (between). Volume-specific reliability is within minus between.

    python scripts/diagnostics/label_group_structure_r8.py
"""
import argparse
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.datasets.fastmri_data import condition_name
from activemri.training.builders import build_dataloaders
from label_curriculum_r8 import gaussian_blur
from label_predictor_gonogo_r8 import available_of, label_probability, load_shards
from oracle_labels_r8 import OUT_DIR, RUN

OUT = Path('analysis/label_gonogo/group_structure.json')
PROTOCOLS = ('brain', 'knee-PD', 'knee-PDFS')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--label-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0, 2, 4, 8])
    parser.add_argument('--splits', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def slice_sources(indices):
    """(file, slice, protocol) of each training-set index, in the dataset order oracle_labels_r8.py indexed."""
    cfg = OmegaConf.load(RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 0
    train_loader, _ = build_dataloaders(cfg)
    examples, anatomies = [], []
    for dataset in train_loader.dataset.datasets:
        examples += dataset.examples
        anatomies += [dataset.anatomy] * len(dataset.examples)
    protocol_of = {}
    for path, anatomy in {(e[0], a) for e, a in zip(examples, anatomies)}:
        with h5py.File(path, 'r') as handle:
            protocol_of[path] = condition_name(anatomy, dict(handle.attrs))
    rows = [examples[int(i)] for i in indices]
    return (np.array([str(path) for path, _ in rows]), np.array([int(s) for _, s in rows]),
            np.array([protocol_of[path] for path, _ in rows]))


def density_blur(probability, weight, sigma):
    """Normalised-convolution blur of [N, H, W] probabilities over the locations where weight is 1."""
    if sigma <= 0:
        return probability
    numerator = gaussian_blur((probability * weight).unsqueeze(1), sigma)[:, 0]
    return numerator / gaussian_blur(weight.unsqueeze(1), sigma)[:, 0].clamp(min=1e-6)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def stratified_halves(volumes, groups, rng):
    """Two disjoint volume sets, each holding half of every group's volumes."""
    first = set()
    for group in np.unique(groups):
        members = rng.permutation(volumes[groups == group])
        first |= set(members[:len(members) // 2])
    return first


def protocol_reliability(field, volume, protocol, pair, args, permute):
    """Split-half correlation and noise-free energy of the pair[0]-minus-pair[1] mean label, groups optionally permuted."""
    rng = np.random.RandomState(args.seed)
    keep = np.isin(protocol, pair)
    field, volume, protocol = field[keep], volume[keep], protocol[keep]
    volumes = np.unique(volume)
    volume_protocol = np.array([protocol[volume == v][0] for v in volumes])
    values, energies = [], []
    for _ in range(args.splits):
        assigned = rng.permutation(volume_protocol) if permute else volume_protocol
        slice_protocol = assigned[np.searchsorted(volumes, volume)]
        first = stratified_halves(volumes, assigned, rng)
        in_first = np.isin(volume, list(first))
        differences = [field[half & (slice_protocol == pair[0])].mean(0) - field[half & (slice_protocol == pair[1])].mean(0)
                       for half in (in_first, ~in_first)]
        values.append(corr(*differences))
        energies.append(float((differences[0] * differences[1]).sum()))
    return {'reliability': float(np.mean(values)), 'energy': float(np.mean(energies))}


def explained_share(field, volume, protocol, args, permute, keep=None):
    """Share of the kept labels' variance around their mean explained by the group means."""
    rng = np.random.RandomState(args.seed)
    keep = np.ones(len(protocol), bool) if keep is None else keep
    field, volume, protocol = field[keep], volume[keep], protocol[keep]
    volumes = np.unique(volume)
    volume_protocol = np.array([protocol[volume == v][0] for v in volumes])
    total = float(((field - field.mean(0)) ** 2).sum())
    shares = []
    for _ in range(args.splits if permute else 1):
        assigned = rng.permutation(volume_protocol) if permute else volume_protocol
        slice_protocol = assigned[np.searchsorted(volumes, volume)]
        explained = sum(float((slice_protocol == p).sum()) * float(((field[slice_protocol == p].mean(0) - field.mean(0)) ** 2).sum())
                        for p in np.unique(slice_protocol))
        shares.append(explained / total)
    return float(np.mean(shares))


def volume_reliability(field, volume, slice_index, protocol, split):
    """Within-volume and between-volume correlation of half-volume mean deviations from the protocol mean."""
    halves = {}
    for v in np.unique(volume):
        rows = np.where(volume == v)[0]
        if len(rows) < 4:
            continue
        rows = rows[np.argsort(slice_index[rows])]
        parts = (rows[0::2], rows[1::2]) if split == 'alternate' else (rows[:len(rows) // 2], rows[len(rows) // 2:])
        others = (protocol == protocol[rows[0]]) & (volume != v)
        reference = field[others].mean(0)
        halves[v] = (protocol[rows[0]], [field[part].mean(0) - reference for part in parts])
    within, between = [], []
    names = list(halves)
    for v in names:
        group, (first, second) = halves[v]
        within.append(corr(first, second))
        between += [corr(first, halves[u][1][1]) for u in names if u != v and halves[u][0] == group]
    return {'within': float(np.mean(within)), 'between': float(np.mean(between)),
            'volume_specific': float(np.mean(within) - np.mean(between)), 'volumes': len(within)}


def main():
    args = parse_args()
    torch.set_num_threads(8)
    labels = load_shards(args.label_dir, 'train')
    volume, slice_index, protocol = slice_sources(labels['indices'].numpy())
    assert all(p.startswith(a) for p, a in zip(protocol, labels['anatomy'])), 'label anatomy and file disagree'
    probability = label_probability(labels)[:, 0].float()
    available = available_of(labels)[:, 0]
    common = available.bool().all(0)
    distinct_supports = len({bytes(m.numpy().tobytes()) for m in labels['valid_mask'][:, 0].bool()})
    counts = {p: {'volumes': int(len(np.unique(volume[protocol == p]))), 'labels': int((protocol == p).sum())} for p in PROTOCOLS}
    print('labels per protocol', json.dumps(counts), '| common locations', int(common.sum()), flush=True)

    results = {'counts': counts, 'common_locations': int(common.sum()), 'distinct_valid_masks': distinct_supports, 'by_sigma': {}}
    for sigma in args.sigmas:
        field = density_blur(probability, available, sigma)[:, common]
        anatomy_of = np.where(np.char.startswith(protocol.astype(str), 'knee'), 'knee', 'brain')
        knee = anatomy_of == 'knee'
        row = {
            'brain_vs_knee': protocol_reliability(field, volume, anatomy_of, ('brain', 'knee'), args, False),
            'pd_vs_pdfs': protocol_reliability(field, volume, protocol, ('knee-PD', 'knee-PDFS'), args, False),
            'pd_vs_pdfs_null': protocol_reliability(field, volume, protocol, ('knee-PD', 'knee-PDFS'), args, True),
            'knee_explained_share': explained_share(field, volume, protocol, args, False, knee),
            'knee_explained_share_null': explained_share(field, volume, protocol, args, True, knee),
            'anatomy_explained_share': explained_share(field, volume, anatomy_of, args, False),
            'anatomy_explained_share_null': explained_share(field, volume, anatomy_of, args, True),
            'volume_alternate': volume_reliability(field, volume, slice_index, protocol, 'alternate'),
            'volume_blocks': volume_reliability(field, volume, slice_index, protocol, 'blocks'),
        }
        row['energy_ratio_pd_pdfs_to_brain_knee'] = row['pd_vs_pdfs']['energy'] / row['brain_vs_knee']['energy']
        results['by_sigma'][str(sigma)] = row
        print(f'sigma {sigma:g}', json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results['config'] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    args.out.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
