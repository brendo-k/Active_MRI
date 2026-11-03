"""Check that the trainer throughput switches leave the training math unchanged (reports/varnet_train_speed).

Train mode trains 2 short epochs and saves the parameters, the epoch losses and the CPU/CUDA RNG states. It is
deterministic: 2 cascades, no compile, cuDNN autotuner off, deterministic algorithms where available, batch 8,
limit_batches 0.1. Compare mode checks two such files bitwise and writes a JSON verdict.

    CUDA_VISIBLE_DEVICES=N python scripts/diagnostics/varnet_train_speed_equivalence.py train <repo> a.pt
    CUDA_VISIBLE_DEVICES=N python scripts/diagnostics/varnet_train_speed_equivalence.py train <repo> b.pt \
        training.train_metrics_every=10 training.visualize_every=5
    python scripts/diagnostics/varnet_train_speed_equivalence.py compare a.pt b.pt out.json
"""
import json
import os
import sys

if sys.argv[1] == 'compare':
    import torch
    a, b = (torch.load(path, weights_only=False) for path in sys.argv[2:4])
    result = {
        'a_overrides': a['extra'], 'b_overrides': b['extra'],
        'epoch_losses_a': [float(x) for x in a['losses']], 'epoch_losses_b': [float(x) for x in b['losses']],
        'losses_equal': [float(x) for x in a['losses']] == [float(x) for x in b['losses']],
        'cuda_rng_equal': bool(torch.equal(a['cuda_rng'], b['cuda_rng'])), 'cpu_rng_equal': bool(torch.equal(a['cpu_rng'], b['cpu_rng'])),
        'param_tensors': len(a['params']),
        'params_bitwise_equal': all(torch.equal(a['params'][n], b['params'][n]) for n in a['params']),
        'max_abs_param_diff': max((a['params'][n] - b['params'][n]).abs().max().item() for n in a['params']),
    }
    open(sys.argv[4], 'w').write(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    sys.exit(0)

assert sys.argv[1] == 'train'
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
repo, out = sys.argv[2], sys.argv[3]
extra = sys.argv[4:]
sys.path.insert(0, repo)
os.chdir(repo)

import torch
from hydra import compose, initialize_config_dir

torch.use_deterministic_algorithms(True, warn_only=True)
BASE = ['recon=varnet', 'datasets=fastmri_local_v1_flair_knee', 'misc.resolution=[320,320]', 'sampler.accelerations=[8.0]',
        'sampler.budget_over=acquired', 'training.num_step=1', 'training.batch_size=8', 'training.num_epochs=2',
        'training.lr_step_size=36', 'training.seed=42', 'sampler.sampler_type=loupe_2d', 'sampler.loupe_vd_init_power=4',
        'sampler.anatomy_conditioning=false', 'datasets.limit_volumes=0.2', 'recon.num_cascades=2',
        'training.compile_unets=false', 'training.cudnn_benchmark=false', 'training.limit_batches=0.1',
        f'output.out_dir={os.path.dirname(out)}/equiv_run_{os.path.basename(out)}']
with initialize_config_dir(config_dir=os.path.join(repo, 'configs'), version_base=None):
    cfg = compose(config_name='config', overrides=BASE + extra)

from activemri.training.builders import build_dataloaders, build_train_config
from activemri.training.trainer import Sequential_Trainer
import train as train_module

config = build_train_config(cfg)
train_module.seed_everything(cfg.training.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
train_loader, val_loader = build_dataloaders(cfg)
trainer = Sequential_Trainer(config, train_loader, torch.device('cuda'), val_loader=val_loader)
losses = []
for epoch in range(2):
    trainer.epoch = epoch
    trainer.scheduler.step(epoch)
    loss, _, _ = trainer.train_epoch()
    losses.append(loss)
state = {name: p.detach().double().sum().item() for name, p in trainer.model.named_parameters()}
torch.save({'losses': losses, 'param_sums': state, 'cuda_rng': torch.cuda.get_rng_state(), 'cpu_rng': torch.get_rng_state(),
            'params': {n: p.detach().cpu() for n, p in trainer.model.named_parameters()}, 'extra': extra}, out)
print('losses', losses)
