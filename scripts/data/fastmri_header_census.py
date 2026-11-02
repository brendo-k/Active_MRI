"""Header-only census of fastMRI multicoil h5 files: one JSON row per volume (no k-space is read).

Standalone (h5py + stdlib) so it runs on CoreHPC, where the repo copy differs. Usage:
    python fastmri_header_census.py --out census.json DIR_OR_GLOB [DIR_OR_GLOB ...]
"""
import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET

import h5py

NS = {'m': 'http://www.ismrm.org/ISMRMRD'}


def _text(root, path, cast=str):
    node = root.find(path, NS)
    return None if node is None or node.text is None else cast(node.text)


def _space(root, name):
    base = f'm:encoding/m:{name}'
    return (
        [_text(root, f'{base}/m:matrixSize/m:{a}', int) for a in 'xyz'],
        [_text(root, f'{base}/m:fieldOfView_mm/m:{a}', float) for a in 'xyz'],
    )


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def census_row(path: str) -> dict:
    with h5py.File(path, 'r') as data:
        attrs = {k: _decode(v) for k, v in data.attrs.items()}
        shape = list(data['kspace'].shape)
        root = ET.fromstring(data['ismrmrd_header'][()])
        keys = sorted(data.keys())
    enc_matrix, enc_fov = _space(root, 'encodedSpace')
    rec_matrix, rec_fov = _space(root, 'reconSpace')
    limits = 'm:encoding/m:encodingLimits/m:kspace_encoding_step_1'
    system = 'm:acquisitionSystemInformation'
    sequence = 'm:sequenceParameters'
    return {
        'file': os.path.basename(path),
        'dir': os.path.dirname(path),
        'bytes': os.stat(path).st_size,
        'keys': keys,
        'acquisition': str(attrs.get('acquisition', '')),
        'patient_id': str(attrs.get('patient_id', '')),
        'kspace_shape': shape,
        'vendor': _text(root, f'{system}/m:systemVendor'),
        'model': _text(root, f'{system}/m:systemModel'),
        'field_T': _text(root, f'{system}/m:systemFieldStrength_T', float),
        'channels': _text(root, f'{system}/m:receiverChannels', int),
        'coil_name': _text(root, f'{system}/m:coilLabel/m:coilName'),
        'institution': _text(root, f'{system}/m:institutionName'),
        'enc_matrix': enc_matrix,
        'enc_fov_mm': enc_fov,
        'rec_matrix': rec_matrix,
        'rec_fov_mm': rec_fov,
        'pe_min': _text(root, f'{limits}/m:minimum', int),
        'pe_max': _text(root, f'{limits}/m:maximum', int),
        'pe_center': _text(root, f'{limits}/m:center', int),
        'accel': _text(root, 'm:encoding/m:parallelImaging/m:accelerationFactor/m:kspace_encoding_step_1', int),
        'TR': _text(root, f'{sequence}/m:TR', float),
        'TE': _text(root, f'{sequence}/m:TE', float),
        'TI': _text(root, f'{sequence}/m:TI', float),
        'flip': _text(root, f'{sequence}/m:flipAngle_deg', float),
        'sequence': _text(root, f'{sequence}/m:sequence_type'),
        'echo_spacing': _text(root, f'{sequence}/m:echo_spacing', float),
        'protocol': _text(root, 'm:measurementInformation/m:protocolName'),
        'patient_position': _text(root, 'm:measurementInformation/m:patientPosition'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', help='directories (all *.h5 inside) or glob patterns')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    paths = []
    for source in args.sources:
        paths += sorted(glob.glob(os.path.join(source, '*.h5'))) if os.path.isdir(source) else sorted(glob.glob(source))
    rows, failures = [], []
    for i, path in enumerate(paths):
        try:
            rows.append(census_row(path))
        except Exception as err:  # unreadable or truncated files are reported, not fatal
            failures.append({'file': path, 'error': repr(err)})
        if (i + 1) % 500 == 0:
            print(f'{i + 1}/{len(paths)}', flush=True)
    with open(args.out, 'w') as f:
        json.dump({'rows': rows, 'failures': failures}, f)
    print(f'wrote {len(rows)} rows, {len(failures)} failures to {args.out}')


if __name__ == '__main__':
    main()
