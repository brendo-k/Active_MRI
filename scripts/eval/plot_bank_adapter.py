"""How the `bank_adapt_2d` gate routes between the bank's masks on the validation set.

Loads a bank run's checkpoint, runs `propose_mask` over evenly spaced val slices, hooks the gate's softmax weights, and plots:
the bank's masks, the mean gate weight per class, the spread of the weights, the mean blended mask per class, and example slices.

    python scripts/eval/plot_bank_adapter.py RUN_DIR --out-dir DIR [--checkpoint last_model.pt] [--limit 900]
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import build_model
from activemri.utils.mask_binarization import floored_grad_sigmoid

torch.multiprocessing.set_start_method('fork', force=True)  # the default forkserver re-runs this module in every DataLoader worker

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
parser.add_argument('--checkpoint', default='last_model.pt')
parser.add_argument('--limit', type=int, default=900)
parser.add_argument('--chunk', type=int, default=8)
parser.add_argument('--out-dir', type=Path, required=True)
args = parser.parse_args()

checkpoint = torch.load(args.run / args.checkpoint, map_location='cpu', weights_only=False)
model = build_model(checkpoint['options'])
model.load_state_dict(checkpoint['model'])
model.eval().requires_grad_(False)
bank = model.sampler.mask_net
K = bank.bank_size
epoch = checkpoint.get('epoch', -1) + 1
print(f'{args.run.name}: {args.checkpoint}, epoch {epoch}, bank_size {K}, k-space grid {bank.shape}', flush=True)

cfg = OmegaConf.load(args.run / '.hydra' / 'config.yaml')
cfg.datasets.num_workers = 4
_, val_loader = build_dataloaders(cfg)
dataset = val_loader.dataset
idx = np.linspace(0, len(dataset) - 1, min(args.limit, len(dataset))).astype(int).tolist()
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, idx), batch_size=args.chunk, shuffle=False,
                                     num_workers=4, collate_fn=val_loader.collate_fn)

captured = []
bank.gate_head.register_forward_hook(lambda m, i, o: captured.append(torch.softmax(o, dim=-1).detach().cpu()))
weights, blended, targets, condition, fnames, slice_ids = [], [], [], [], [], []
with torch.no_grad():
    for n, batch in enumerate(loader):
        proposal = model.propose_mask(batch['kspace'])
        weights.append(captured.pop())
        blended.append(proposal['unscaled'][:, 0].cpu())
        targets.append(batch['target'].reshape(len(batch['kspace']), *batch['target'].shape[-2:]).cpu())
        condition += list(batch['condition'])
        fnames += list(batch['fname'])
        slice_ids += [int(s) for s in batch['slice_id']]
        if n % 10 == 0:
            print(f'{n * args.chunk}/{len(idx)}', flush=True)
W = torch.cat(weights).numpy()  # [N, K]
P = torch.cat(blended).numpy()  # [N, H, W]
T = torch.cat(targets).numpy()
condition = np.array(condition)
classes = list(dict.fromkeys(condition))
sig = floored_grad_sigmoid(bank.mask, bank.slope, bank.grad_floor).detach().numpy()  # [K, H, W]
shift = lambda a: np.fft.fftshift(a, axes=(-2, -1))  # masks are stored with DC at the corner; show them centred

entropy = -(W * np.log(W + 1e-12)).sum(1)
print(f'{len(W)} slices; effective number of masks per slice exp(H): mean {np.exp(entropy).mean():.2f} (uniform would be {K})')
print('max weight per slice: mean %.3f, 5-95%%: %.3f-%.3f' % (W.max(1).mean(), *np.percentile(W.max(1), [5, 95])))
print('mean sampled probability of each bank mask (outside the centre block excluded not applied):', np.round(sig.mean((1, 2)), 3))
for c in classes:
    m = condition == c
    print(f'{c:10s} n={m.sum():4d}  mean weights {np.round(W[m].mean(0), 3)}  top mask {W[m].mean(0).argmax()}  exp(H) {np.exp(entropy[m]).mean():.2f}')

fig = plt.figure(figsize=(20, 17))
gs = fig.add_gridspec(5, 8, hspace=0.45, wspace=0.15)
for k in range(K):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(shift(sig[k]), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'bank mask {k}\nmean p {sig[k].mean():.2f}', fontsize=8)
    ax.axis('off')
ax = fig.add_subplot(gs[1, :3])
means = np.stack([W[condition == c].mean(0) for c in classes])
im = ax.imshow(means, cmap='viridis', vmin=0, aspect='auto')
ax.set_xticks(range(K)); ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
ax.set_xlabel('bank mask'); ax.set_title(f'mean gate weight per class (uniform = {1 / K:.3f})')
for i in range(len(classes)):
    for j in range(K):
        ax.text(j, i, f'{means[i, j]:.2f}', ha='center', va='center', color='w', fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.03)
ax = fig.add_subplot(gs[1, 3:6])
for c, colour in zip(classes, ('C0', 'C1', 'C2', 'C3')):
    m = condition == c
    ax.boxplot([W[m][:, k] for k in range(K)], positions=np.arange(K) + 0.25 * classes.index(c) - 0.25, widths=0.2,
               patch_artist=True, boxprops=dict(facecolor=colour, alpha=0.6), showfliers=False)
ax.set_xticks(range(K)); ax.set_xlabel('bank mask'); ax.set_ylabel('gate weight'); ax.set_title('per-slice weight spread (colours = classes: ' + ', '.join(f'C{i}={c}' for i, c in enumerate(classes)) + ')', fontsize=8)
ax = fig.add_subplot(gs[1, 6:])
ax.hist(np.exp(entropy), bins=40, color='gray'); ax.axvline(K, color='r', ls='--', label='uniform')
ax.set_xlabel('exp(entropy of gate weights)  = effective number of masks'); ax.legend(); ax.set_title('how peaked the gate is')
for r, c in enumerate(classes):
    m = condition == c
    ax = fig.add_subplot(gs[2, 2 * r:2 * r + 2])
    ax.imshow(shift(P[m].mean(0)), cmap='gray', vmin=0, vmax=1); ax.set_title(f'mean blended mask, {c}', fontsize=9); ax.axis('off')
if len(classes) >= 2:
    ax = fig.add_subplot(gs[2, 6:])
    d = shift(P[condition == classes[0]].mean(0) - P[condition == classes[1]].mean(0))
    v = np.abs(d).max()
    ax.imshow(d, cmap='RdBu_r', vmin=-v, vmax=v); ax.set_title(f'mean blended mask: {classes[0]} minus {classes[1]}', fontsize=9); ax.axis('off')
rng = np.random.RandomState(0)
ex = {c: rng.choice(np.flatnonzero(condition == c), 2, replace=False) for c in classes}
for r, c in enumerate(classes):
    for j, i in enumerate(ex[c]):
        a0 = fig.add_subplot(gs[3, r * 2 + j])
        a0.imshow(T[i], cmap='gray'); a0.set_title(f'{c} {fnames[i][-18:-3]} s{slice_ids[i]}', fontsize=6); a0.axis('off')
        a1 = fig.add_subplot(gs[4, r * 2 + j])
        a1.bar(range(K), W[i]); a1.set_ylim(0, max(0.5, W[i].max() * 1.1)); a1.axhline(1 / K, color='r', lw=0.5)
        a1.set_title('gate weights', fontsize=6); a1.tick_params(labelsize=6)
fig.suptitle(f'{args.run.name} ({args.checkpoint}, epoch {epoch}): {K}-mask bank, {len(W)} val slices', fontsize=13)
args.out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out_dir / f'{args.run.name}_bank_adapter.png', dpi=75, bbox_inches='tight')
np.savez(args.out_dir / f'{args.run.name}_bank_weights.npz', weights=W, condition=condition, fnames=np.array(fnames), slice_ids=np.array(slice_ids), epoch=epoch)
print('saved')
