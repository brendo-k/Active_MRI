"""Preprocess a Siemens 3D Cartesian twix file into the h5 `SiemensSpaceDataset` reads.

    python scripts/siemens_space/preprocess.py meas_MID00295_FID30725_bk_space.dat --grid 256 128
"""
import _path  # noqa: F401
import argparse
import pathlib

from activemri.datasets.siemens_space_preprocess import preprocess

SCANS_DIR = pathlib.Path('/export/scratch/bkadota')
CACHE_DIR = SCANS_DIR / 'space_cache'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dat', help='twix file name inside the scans dir, or a path')
    parser.add_argument('--grid', type=int, nargs=2, required=True, metavar=('KY', 'KZ'),
                        help='k-space grid (even sizes); must hold every acquired line around its centre')
    parser.add_argument('--out-dir', type=pathlib.Path, default=CACHE_DIR)
    args = parser.parse_args()

    dat = pathlib.Path(args.dat)
    dat = dat if dat.exists() else SCANS_DIR / dat
    out = preprocess(dat, args.out_dir / f'{dat.stem}.h5', tuple(args.grid))
    print(out)


if __name__ == '__main__':
    main()
