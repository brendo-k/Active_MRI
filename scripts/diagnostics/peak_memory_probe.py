"""Run train.py for a few training batches, print peak GPU memory, and exit before validation.

    python scripts/diagnostics/peak_memory_probe.py <the train.py arguments> training.limit_batches=0.01
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train
from activemri.training import trainer


def report(device):
    props = torch.cuda.get_device_properties(device)
    print(f'PROBE gpu={props.name} total={props.total_memory / 2**30:.1f}GiB '
          f'peak_allocated={torch.cuda.max_memory_allocated(device) / 2**30:.1f}GiB '
          f'peak_reserved={torch.cuda.max_memory_reserved(device) / 2**30:.1f}GiB', flush=True)


original = trainer.Sequential_Trainer.train_epoch


def train_epoch(self):
    start = time.perf_counter()
    try:
        original(self)
    except torch.cuda.OutOfMemoryError:
        print('PROBE out of memory', flush=True)
        report(self.options.device)
        raise
    print(f'PROBE train_epoch {time.perf_counter() - start:.1f}s', flush=True)
    report(self.options.device)
    sys.exit(0)


trainer.Sequential_Trainer.train_epoch = train_epoch
train.main()
