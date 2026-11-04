"""One real batch at full scale through the Muon + LayerNorm control: eager coil ops against the fused path, same weights,
same batch, same mask draw, in one process. Relative differences of the output and of the gradients per model part, against
the same eager pass run twice. Needs a GPU.

    CUDA_VISIBLE_DEVICES=N python scripts/diagnostics/muon_layer_speed_gradcheck.py <out.json> [hydra overrides...]

With fp32 U-Nets and TF32 off (MLS_NO_TF32=1) there is no bf16 rounding to amplify, so what remains is the float32 rounding of
the coil arithmetic itself. L1's gradient is sign(output - target), discontinuous at pixels with a near-zero residual, so
MLS_SMOOTH_LOSS=1 swaps it for a mean square to compare the gradients without those sign flips.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('WANDB_MODE', 'disabled')
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import torch
from hydra import compose, initialize_config_dir

from activemri.training.builders import build_dataloaders, build_train_config
from activemri.training.trainer import Sequential_Trainer
import train as train_module

CONTROL = ['recon=varnet', 'datasets=fastmri_local_v1_flair_knee', 'misc.resolution=[320,320]', 'sampler.accelerations=[8.0]',
           'sampler.budget_over=acquired', 'training.num_step=1', 'training.batch_size=16', 'training.num_epochs=100',
           'training.lr_step_size=90', 'training.seed=42', 'sampler.sampler_type=loupe_2d', 'sampler.loupe_vd_init_power=4',
           'sampler.anatomy_conditioning=false', 'datasets.limit_volumes=0.05', 'training.optimizer=muon', 'recon.norm=layer',
           'sampler.sens_norm=layer', 'training.cudnn_benchmark=false']
META = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0}
PARTS = ('reconstructor', 'sensetivity_net', 'sampler')

torch.backends.cudnn.deterministic = True
if os.environ.get('MLS_NO_TF32'):
    torch.backends.cudnn.allow_tf32 = False
with initialize_config_dir(config_dir=str(REPO / 'configs'), version_base=None):
    cfg = compose(config_name='config', overrides=CONTROL + sys.argv[2:] + [f'output.out_dir={sys.argv[1]}.work'])
config = build_train_config(cfg)
train_module.seed_everything(cfg.training.seed)
train_loader, val_loader = build_dataloaders(cfg)
trainer = Sequential_Trainer(config, train_loader, torch.device('cuda'), val_loader=val_loader)
model = trainer.model
model.train()
model.store_visuals = False
batch = next(iter(train_loader))


def gradients(fused):
    model.reconstructor.use_fused_coil_ops = fused
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    kspace, target = batch['kspace'].cuda(), batch['target'].cuda()
    pred = model(target, kspace, **trainer._model_inputs(batch))
    loss = model.loss(pred, trainer._target_dict(batch, target, kspace), META, config.loss_type)[0]
    if os.environ.get('MLS_SMOOTH_LOSS'):
        loss = (pred['output'] - target.unsqueeze(1)).square().mean()
    trainer.optimizer.zero_grad()
    loss.backward()
    grads = {part: torch.cat([p.grad.flatten() for p in getattr(model, part).parameters() if p.grad is not None]).clone() for part in PARTS}
    return {'output': pred['output'].detach().clone(), **grads}


relative = lambda a, b: ((a - b).norm() / b.norm()).item()
eager, eager_again, fused = gradients(False), gradients(False), gradients(True)
result = {key: {'fused_vs_eager': relative(fused[key], eager[key]), 'eager_vs_eager': relative(eager_again[key], eager[key])} for key in eager}
Path(sys.argv[1]).write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
