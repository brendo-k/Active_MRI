#!/bin/bash
# Header census of CoreHPC's fastMRI knee (train+val) and brain AXT2/AXFLAIR (train+val).
# Run from the folder holding fastmri_header_census.py: sbatch fastmri_header_census_corehpc.sbatch
#SBATCH --job-name=fastmri_census
#SBATCH --partition=cpu
#SBATCH --account=hpc_core
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail
source /mnt/software/ohpc/pub/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate dl
python -c "import h5py; print('h5py', h5py.__version__)"

data=~/scratch/data/fastmri
python fastmri_header_census.py --out census_knee.json \
    "$data/knee/data/multicoil_train" "$data/knee/data/multicoil_val"
python fastmri_header_census.py --out census_brain.json \
    "$data/brain/data/multicoil_train/file_brain_AXT2_*.h5" "$data/brain/data/multicoil_train/file_brain_AXFLAIR_*.h5" \
    "$data/brain/data/multicoil_val/file_brain_AXT2_*.h5" "$data/brain/data/multicoil_val/file_brain_AXFLAIR_*.h5"
