"""Is the ACS gate's coil-count effect separable from the scanner? (CPU only)

For every brain volume of the balanced split (train and val), reads the fastMRI ISMRMRD header (system model, field
strength, receiver channels) and the raw coil count of the k-space, and joins the validation volumes with their
held-out gain of the per-volume grid over its control (`acs_gate_theta_check_r8.py --direction gain` output).

    python scripts/diagnostics/acs_gate_scanner_check_r8.py
"""
import json
import pathlib
import xml.etree.ElementTree as ElementTree
from collections import Counter

import h5py
import numpy as np

import _path  # noqa: F401
from activemri.datasets.fastmri_data import _build_balanced_datasets

GATE = pathlib.Path('analysis/acs_gate_cgsense')
NAMESPACE = {'m': 'http://www.ismrm.org/ISMRMRD'}
FEW_COILS = 8


def header(path):
    with h5py.File(path, 'r', locking=False) as f:
        root = ElementTree.fromstring(f['ismrmrd_header'][()])
        coils = int(f['kspace'].shape[1])
    system = root.find('m:acquisitionSystemInformation', NAMESPACE)
    text = lambda tag: system.findtext(f'm:{tag}', default='', namespaces=NAMESPACE)
    return {'system_model': text('systemModel'), 'field_strength_T': float(text('systemFieldStrength_T') or 'nan'),
            'receiver_channels': int(text('receiverChannels') or -1), 'kspace_coils': coils,
            'scanner_code': pathlib.Path(path).name.split('_')[3]}


def main():
    train, val = _build_balanced_datasets(pathlib.Path('/home/bkadota/data/fastmri'), ['brain', 'knee'], 'T2', 0.2,
                                          image_shape=[256, 256], transform=None, num_coils=16, min_energy_fraction=0.2,
                                          balance_by='anatomy')
    gain = {v['volume']: v['held_out_gain'] for v in json.loads((GATE / 'theta_check_gain.json').read_text())['per_volume']}
    result = {}
    for split, datasets in (('train', train), ('val', val)):
        brain = next(d for d in datasets if d.anatomy == 'brain')
        volumes = sorted({path for path, _ in brain.examples})
        rows = []
        for path in volumes:
            row = {'volume': pathlib.Path(path).name, **header(path)}
            if split == 'val':
                row['held_out_gain'] = gain.get(row['volume'])
            rows.append(row)
        result[split] = rows
        few = [r for r in rows if r['kspace_coils'] <= FEW_COILS]
        print(f'\n[{split}] {len(rows)} brain volumes, {len(few)} with <= {FEW_COILS} coils')
        groups = Counter((r['system_model'], r['field_strength_T'], r['kspace_coils'] <= FEW_COILS) for r in rows)
        print('| system model | field (T) | <= 8 coils | volumes | mean held-out gain |\n|---|---|---|---|---|')
        for (model, field, is_few), count in sorted(groups.items()):
            gains = [r['held_out_gain'] for r in rows if (r['system_model'], r['field_strength_T'], r['kspace_coils'] <= FEW_COILS)
                     == (model, field, is_few) and r.get('held_out_gain') is not None]
            print(f"| {model} | {field:g} | {is_few} | {count} | {np.mean(gains):+.3f} |" if gains else
                  f"| {model} | {field:g} | {is_few} | {count} | |")
    (GATE / 'scanner_check.json').write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
