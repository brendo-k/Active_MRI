"""Train CG-SENSE knee model with recon input, g-factor conditioning, and channel uncertainty."""
import os
import sys
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import random
from pathlib import Path

import numpy as np
import torch

from activemri.training.trainer import Sequential_Trainer
from activemri.config import TrainConfig
from activemri.reconstructors.config import CGSenseReconConfig
from activemri.datasets.fastmri_data import build_fastmri_dataloaders

SEED = 42

config = TrainConfig(
    model='SequentialSampling',
    num_step=1,
    batch_size=4,
    num_epochs=50,
    lr=0.001,
    lr_step_size=40,
    lr_gamma=0.1,
    weight_decay=0.0,
    loss_type='l1',
    device='cuda',
    out_dir=[Path('checkpoints/knee_cgsense_recon_gfactor_320')],
    accelerations=[4.0],
    line_constrained=False,
    preselect_num=10,
    resolution=[320, 320],
    random_rotate=False,
    # New features
    with_recon=True,
    with_gfactor=True,
    gfactor_replicas=4,
    uncertainty_mode='channel',
    recon=CGSenseReconConfig(
        cg_iters=10,
        cg_lamda=0.0,
        sense_tikhonov_weight=0.0,
    ),
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

train_loader, val_loader = build_fastmri_dataloaders(
    data_root=Path('/home/bkadota/data/fastmri'),
    anatomies=['knee'],
    image_shape=[320, 320],
    batch_size=4,
    num_coils=15,
    num_workers=4,
    min_energy_fraction=0.2,
    data_dir='/home/bkadota/data/fastmri/combined_14',
)

trainer = Sequential_Trainer(config, train_loader, torch.device('cuda'), val_loader=val_loader)
trainer()
