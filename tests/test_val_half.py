"""`datasets.val_half` splits each condition's val volumes into two fixed, disjoint halves (dev to iterate, holdout to
confirm). Run with ``.venv/bin/python -m pytest tests/test_val_half.py`` from the repo root."""
import csv

import pytest

from activemri.datasets.fastmri_data import REPO_ROOT, _val_half_rows, condition_name

MANIFEST = REPO_ROOT / 'configs' / 'datasets' / 'fastmri_local_v1' / 'manifest.csv'


def knee_val_rows():
    with open(MANIFEST) as f:
        return [row for row in csv.DictReader(f) if row['cls'] == 'knee' and row['split'] == 'val']


def files(rows):
    return {row['file'] for row in rows}


def test_dev_and_holdout_are_disjoint_halves_of_each_condition():
    rows = knee_val_rows()
    dev, holdout = _val_half_rows(rows, 'dev'), _val_half_rows(rows, 'holdout')
    assert files(dev).isdisjoint(files(holdout))
    assert files(dev) | files(holdout) == files(rows)
    for condition in ('knee-PD', 'knee-PDFS'):
        count = lambda part: sum(condition_name(r['anatomy'], r) == condition for r in part)
        assert count(dev) == count(holdout) == count(rows) // 2


def test_the_split_is_fixed_and_independent_of_other_conditions():
    rows = knee_val_rows()
    pd_only = [r for r in rows if condition_name(r['anatomy'], r) == 'knee-PD']
    assert files(_val_half_rows(rows, 'dev')) == files(_val_half_rows(list(reversed(rows)), 'dev'))
    pd_dev = {r['file'] for r in _val_half_rows(rows, 'dev') if condition_name(r['anatomy'], r) == 'knee-PD'}
    assert files(_val_half_rows(pd_only, 'dev')) == pd_dev


def test_all_keeps_every_row_and_unknown_halves_are_refused():
    rows = knee_val_rows()
    assert _val_half_rows(rows, 'all') == rows
    with pytest.raises(ValueError):
        _val_half_rows(rows, 'test')
