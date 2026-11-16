"""Manifest train tiers: S inside L inside F, F adds FLAIR only, and val/test never change with the tier.

Run with ``.venv/bin/python -m pytest tests/test_train_tiers.py`` from the repo root.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.datasets.fastmri_data import _manifest_split_rows  # noqa: E402

MANIFEST = Path(__file__).resolve().parent.parent / 'configs/datasets/fastmri_local_v1/manifest.csv'


def _rows():
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))


def _train_files(rows, tier, cls):
    train, _ = _manifest_split_rows([r for r in rows if r['cls'] == cls], train_tier=tier)
    return {r['file'] for r in train}


def test_tiers_nest_and_count():
    rows = _rows()
    for cls in ('FLAIR', 'T2', 'knee'):
        s, l, f = (_train_files(rows, tier, cls) for tier in 'SLF')
        assert s < l <= f
        assert (len(s), len(l)) == (80, 300)
    assert len(_train_files(rows, 'F', 'FLAIR')) == 339
    assert _train_files(rows, 'F', 'T2') == _train_files(rows, 'L', 'T2')
    assert _train_files(rows, 'F', 'knee') == _train_files(rows, 'L', 'knee')


def test_tier_f_keeps_val_and_has_no_val_or_test_patients():
    rows = _rows()
    val = {tier: {r['file'] for r in _manifest_split_rows(rows, train_tier=tier, val_limit_volumes=0.25)[1]}
           for tier in 'SLF'}
    assert val['S'] == val['L'] == val['F']
    held_out = {r['patient_id'] for r in rows if r['split'] in ('val', 'test')}
    assert not held_out & {r['patient_id'] for r in rows if r['tier'] == 'F-only'}
