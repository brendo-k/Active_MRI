"""datasets.limit_volumes keeps the same fraction of each condition on every run.

Run with ``.venv/bin/python -m pytest tests/test_limit_volumes.py`` from the repo root.
"""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.datasets.fastmri_data import (  # noqa: E402
    LIMIT_VOLUMES_SEED,
    _limit_volume_rows,
    _manifest_split_rows,
    condition_name,
)

MANIFEST = Path(__file__).resolve().parent.parent / 'configs/datasets/fastmri_local_v1/manifest.csv'


def _row(name, anatomy, acquisition, split='train'):
    return {'file': name, 'anatomy': anatomy, 'acquisition': acquisition, 'split': split}


def _names(rows):
    return [row['file'] for row in rows]


def _by_condition(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(condition_name(row['anatomy'], row), set()).add(row['file'])
    return grouped


def _counts(rows):
    return {condition: len(names) for condition, names in _by_condition(rows).items()}


def test_same_rows_every_call_and_input_order_does_not_matter():
    rows = [_row(f'f{i}.h5', 'brain', 'AXFLAIR') for i in range(10)]
    rows += [_row(f'k{i}.h5', 'knee', 'CORPD_FBK' if i < 6 else 'CORPDFS_FBK') for i in range(12)]
    first = _names(_limit_volume_rows(rows, 0.5))
    second = _names(_limit_volume_rows(list(reversed(rows)), 0.5))
    assert first == second
    assert _counts(_limit_volume_rows(rows, 0.5)) == {'brain': 5, 'knee-PD': 3, 'knee-PDFS': 3}


def test_smaller_fraction_is_contained_in_a_larger_one():
    rows = [_row(f'f{i:02d}.h5', 'brain', 'AXFLAIR') for i in range(40)]
    kept = set()
    for fraction in (0.05, 0.1, 0.25, 0.5, 0.75, 1):
        names = set(_names(_limit_volume_rows(rows, fraction)))
        assert kept <= names
        kept = names
    assert len(kept) == 40


def test_one_condition_does_not_move_another():
    brain = [_row(f'f{i}.h5', 'brain', 'AXFLAIR') for i in range(6)]
    knee = [_row(f'k{i}.h5', 'knee', 'CORPD_FBK') for i in range(6)]
    brain_only = set(_names(_limit_volume_rows(brain, 0.5)))
    brain_with_knee = _by_condition(_limit_volume_rows(brain + knee, 0.5))['brain']
    assert brain_only == brain_with_knee


def test_one_keeps_every_row_and_a_tiny_fraction_keeps_one():
    rows = [_row(f'f{i}.h5', 'brain', 'AXFLAIR') for i in range(8)]
    assert _names(_limit_volume_rows(rows, None)) == _names(rows)
    assert _names(_limit_volume_rows(rows, 1)) == _names(rows)
    assert len(_limit_volume_rows(rows, 0.001)) == 1


@pytest.mark.parametrize('fraction', [0, -0.1, 1.1])
def test_rejects_a_fraction_outside_the_unit_interval(fraction):
    rows = [_row('f0.h5', 'brain', 'AXFLAIR')]
    with pytest.raises(ValueError):
        _limit_volume_rows(rows, fraction)


def test_manifest_tier_s_tenth_is_stable_per_condition():
    with MANIFEST.open() as handle:
        rows = [row for row in csv.DictReader(handle)
                if row['cls'] in ('FLAIR', 'knee') and row['split'] == 'train' and row['tier'] == 'S']
    limited = _limit_volume_rows(rows, 0.1)
    again = _limit_volume_rows(rows, 0.1, seed=LIMIT_VOLUMES_SEED)
    assert _names(limited) == _names(again)
    assert _counts(limited) == {'brain': 8, 'knee-PD': 4, 'knee-PDFS': 4}


def _manifest_rows():
    with MANIFEST.open() as handle:
        return [row for row in csv.DictReader(handle) if row['cls'] in ('FLAIR', 'knee')]


def test_val_anatomies_keep_only_brain_val_and_leave_train_alone():
    rows = _manifest_rows()
    train_all, val_all = _manifest_split_rows(rows, train_anatomies=['brain'])
    train, val = _manifest_split_rows(rows, train_anatomies=['brain'], val_anatomies=['brain'])
    assert _names(train) == _names(train_all)
    assert {row['anatomy'] for row in val} == {'brain'}
    assert _names(val) == [row['file'] for row in val_all if row['anatomy'] == 'brain']


@pytest.mark.parametrize('limit', [0.25, 0.5, 1])
def test_pinned_val_fraction_gives_one_val_set_for_every_train_fraction(limit):
    rows = _manifest_rows()
    _, pinned = _manifest_split_rows(rows, train_anatomies=['brain'], limit_volumes=0.25, val_anatomies=['brain'])
    train, val = _manifest_split_rows(rows, train_anatomies=['brain'], limit_volumes=limit, val_anatomies=['brain'],
                                      val_limit_volumes=0.25)
    assert _names(val) == _names(pinned)
    assert len(train) == round(80 * limit)


def test_unpinned_val_fraction_follows_limit_volumes_as_before():
    rows = _manifest_rows()
    train, val = _manifest_split_rows(rows, limit_volumes=0.5)
    assert _names(train) == _names(_limit_volume_rows([r for r in rows if r['split'] == 'train' and r['tier'] == 'S'], 0.5))
    assert _names(val) == _names(_limit_volume_rows([r for r in rows if r['split'] == 'val'], 0.5))
