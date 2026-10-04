#!/bin/bash
export PYTHONPATH=/home/bkadota/code/active_mri/SeqMRI:$PYTHONPATH
cd /home/bkadota/code/active_mri/SeqMRI
python scripts/train_ema_ddqn.py "$@"
