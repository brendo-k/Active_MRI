"""Checks on a real MNetTrainer batch: a rejected label leaves the reconstructor and its Adam state
bit-identical, and the new locations equal the budget exactly."""
import os
os.environ['WANDB_MODE'] = 'offline'
import copy
import sys

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, os.getcwd())
from activemri.training.builders import build_dataloaders, build_train_config
from activemri.training.mnet_trainer import MNetTrainer

PRETRAINED = 'checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05/best_model.pt'
with initialize_config_dir(os.path.join(os.getcwd(), 'configs'), version_base=None):
    cfg = compose('config', overrides=[
        'recon=varnet', 'datasets=brain_knee_balanced', 'sampler.sampler_type=mnet_2d', 'training.batch_size=4',
        'sampler.accelerations=[8.0]', 'sampler.mnet_label_steps=3', 'sampler.mnet_label_lr=0.05',
        'sampler.mnet_accept=q_check', f'output.pretrained_recon={PRETRAINED}', 'output.out_dir=analysis/mnet_smoke/run',
        'datasets.num_workers=2'])
config = build_train_config(cfg)
train_loader, val_loader = build_dataloaders(cfg)
trainer = MNetTrainer(config, train_loader, torch.device('cuda'), val_loader=val_loader)
trainer.model.train()
batch = next(iter(train_loader))
kspace = trainer._load_kspace(batch)
target = trainer._target(batch, kspace)

# prime Adam state with one accepted-style update so the rollback has state to restore
trainer.options.mnet_accept = 'always'
trainer._optimize(None, batch, kspace, target, 0)

def state():
    params = [p.detach().clone() for p in trainer.recon_params]
    opt = {i: {k: v.clone() if torch.is_tensor(v) else v for k, v in trainer.optimizer.state[p].items()}
           for i, p in enumerate(trainer.recon_params) if p in trainer.optimizer.state}
    return params, opt

before = state()
snapshot = trainer._snapshot()
trainer.options.mnet_accept = 'q_check'
# force a reject: make the quality check impossible to pass
import activemri.training.mnet_trainer as mt
mt.accept_label = lambda *a, **k: False
stats = trainer._optimize(None, batch, kspace, target, 1)
after = state()
assert stats['mnet/accepted'] == 0.0
assert all(torch.equal(a, b) for a, b in zip(before[0], after[0])), 'recon weights changed after reject'
assert before[1].keys() == after[1].keys()
for i in before[1]:
    for k, v in before[1][i].items():
        w = after[1][i][k]
        assert torch.equal(v, w) if torch.is_tensor(v) else v == w, f'adam state {i}.{k} changed'
print('rollback bit-identical OK; flip_frac', stats['mnet/flip_frac'])

with torch.no_grad():
    prop = trainer.model.propose_mask(kspace, trainer._anatomy_labels(batch))
new = (prop['mask'] - prop['old_mask']).flatten(1).sum(1)
print('new locations per slice', new.tolist(), 'budget', trainer.budget_count)
assert (new == trainer.budget_count).all()
