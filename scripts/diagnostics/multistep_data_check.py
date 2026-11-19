"""Check that a machine's copy of the fastMRI local v1 FLAIR + knee data serves the slices the lab server serves.

The manifest loader skips a volume it cannot open with only a warning, so a missing or truncated file would silently
shrink the train or val set. This lists every (volume, slice) the train and val loaders serve, per condition, and
compares the lists with a reference written on the lab server. Trailing key=value arguments are Hydra overrides, such
as the CoreHPC data roots.

    python scripts/diagnostics/multistep_data_check.py --write reference.json
    python scripts/diagnostics/multistep_data_check.py --check reference.json \
        'datasets.roots.hpc="~/scratch/data/fastmri/{anatomy}/data"' datasets.roots.local_batch2=~/scratch/data/fastmri/knee/data
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import _path  # noqa: F401

from hydra import compose, initialize_config_dir  # noqa: E402

from activemri.datasets.fastmri_data import condition_name  # noqa: E402
from activemri.training.builders import build_manifest_loaders  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATASET_OVERRIDES = ['datasets=fastmri_local_v1_flair_knee', 'misc.resolution=[320,320]', 'datasets.limit_volumes=1',
                     'datasets.num_workers=0']


def served_slices(overrides):
    with initialize_config_dir(version_base=None, config_dir=str(REPO / 'configs')):
        cfg = compose(config_name='config', overrides=DATASET_OVERRIDES + overrides)
    rows = {}
    with open(REPO / cfg.datasets.manifest) as f:
        for row in csv.DictReader(f):
            rows[row['file']] = row
    train_loader, val_loader = build_manifest_loaders(cfg, list(cfg.misc.resolution), shuffle_seed=cfg.training.seed)
    summary = {}
    for split, loader in (('train', train_loader), ('val', val_loader)):
        pairs = [(fname.name, int(slice_id)) for dataset in loader.dataset.datasets for fname, slice_id in dataset.examples]
        by_condition = {}
        for name, _ in pairs:
            condition = condition_name(rows[name]['anatomy'], rows[name])
            volumes, slices = by_condition.get(condition, (set(), 0))
            volumes.add(name)
            by_condition[condition] = (volumes, slices + 1)
        summary[split] = {
            'slices': len(pairs),
            'volumes': len({name for name, _ in pairs}),
            'by_condition': {c: [len(v), n] for c, (v, n) in sorted(by_condition.items())},
            'sorted_sha1': hashlib.sha1(json.dumps(sorted(pairs)).encode()).hexdigest(),
            'ordered_sha1': hashlib.sha1(json.dumps(pairs).encode()).hexdigest(),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--write', type=Path, help='write the reference summary here')
    action.add_argument('--check', type=Path, help='compare with this reference summary')
    args, overrides = parser.parse_known_args()
    summary = served_slices(overrides)
    print(json.dumps(summary, indent=2))
    if args.write:
        args.write.write_text(json.dumps(summary, indent=2) + '\n')
        return 0
    reference = json.loads(args.check.read_text())
    failures = [f'{split}.{key}: {summary[split][key]} != {reference[split][key]}'
                for split in ('train', 'val') for key in ('slices', 'volumes', 'by_condition', 'sorted_sha1')
                if summary[split][key] != reference[split][key]]
    for split in ('train', 'val'):
        same_order = summary[split]['ordered_sha1'] == reference[split]['ordered_sha1']
        print(f'{split}: loader order {"matches" if same_order else "differs from"} the reference')
    print('\n'.join(failures) if failures else 'data check passed: the same slices as the reference')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
