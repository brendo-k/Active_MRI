# Deleted ~/data/fastmri trees (2026-10-27)

These listings record the old `/home/bkadota/data/fastmri` layout, which was deleted on 2026-10-27. The local runs before 2026-10-26 trained and validated on it, including `reports/ismrm2027`.

Before the deletion, these configs read it: `brain.yaml`, `knee.yaml`, `brain_knee_balanced.yaml`, `brain_knee_condition_balanced.yaml` and `multi_anatomy.yaml`. They now read the local subset v1 (see `fastmri_local_v1_base.yaml`). The old version of `brain_knee_balanced` is kept as `brain_knee_balanced_legacy.yaml` for CoreHPC. Scripts whose `--data-root` still defaults to `/home/bkadota/data/fastmri` have no data until they are moved to the manifest loader.

| File | Tree | Contents |
|---|---|---|
| `multicoil_brain.tsv` | `multicoil_brain/` | 433 files, 150 GB: `T2/{train,val,test}` and `Flair_activemri_splits/brain_multicoil_{train,val,test}` |
| `multicoil_knee.tsv` | `multicoil_knee/` | 167 files, 57 GB: `No_sat`, `fat_sat` and their `_h5` copies |
| `multicoil_knee_classification.csv` | — | copy of `multicoil_knee/classification.csv` |
| `knee_brain_vd.tsv` | `knee_brain_vd/` | 220 symlinks into the trees above |
| `sense_cache_ecalib.tsv` | `sense_cache_ecalib/` | 162 files, 5 GB of BART ecalib maps made by `scripts/data/build_sense_cache.py` |

Each TSV has the columns `relpath` (relative to the old root), `bytes` and `symlink_target`. The original fastMRI files should still be on CoreHPC under `~/scratch/data/fastmri/{brain,knee}`, so a tree can be rebuilt by pulling the listed file names into the same relative paths.

New data lives under `/export/scratch/bkadota/fastmri_raw/{anatomy}/multicoil_{train,val}`, read through `configs/datasets/fastmri_local_v1/manifest.csv` (see `configs/datasets/fastmri_local_v1_flair_knee.yaml`).
