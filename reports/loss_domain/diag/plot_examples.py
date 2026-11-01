import sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
d = Path = None
from pathlib import Path
d = torch.load(Path(sys.argv[1]) / 'cross_loss_examples.pt', weights_only=False)
conds = d['rss']['cond']
picks = [conds.index(c) for c in ('brain', 'knee-PD', 'knee-PDFS') if c in conds]
print(conds)
fig, axes = plt.subplots(len(picks), 6, figsize=(21, 3.6 * len(picks)))
axes = axes.reshape(len(picks), 6)
for r, i in enumerate(picks):
    t = d['rss']['t'][i]; vmax = float(t.quantile(0.995))
    panels = [('target', t), ('rss out', d['rss']['out'][i]), ('kspace_rss out', d['kspace_rss']['out'][i]),
              ('|ESPIRiT|', d['rss']['xe'][i].abs())]
    for c, (name, img) in enumerate(panels):
        axes[r, c].imshow(img, cmap='gray', vmin=0, vmax=vmax); axes[r, c].set_title(f'{conds[i]} {name}')
    ratio = d['kspace_rss']['out'][i] / t.clamp(min=0.05 * vmax)
    im = axes[r, 4].imshow(ratio, cmap='RdBu_r', vmin=0, vmax=2); axes[r, 4].set_title('kspace_rss out / target'); plt.colorbar(im, ax=axes[r, 4])
    row = t.shape[0] // 2
    axes[r, 5].plot(t[row], label='target', lw=1); axes[r, 5].plot(d['rss']['out'][i][row], label='rss', lw=1)
    axes[r, 5].plot(d['kspace_rss']['out'][i][row], label='kspace_rss', lw=1); axes[r, 5].legend(); axes[r, 5].set_title('centre row profile')
for a in axes[:, :5].flat: a.axis('off')
plt.tight_layout(); plt.savefig(Path(sys.argv[1]) / 'kspace_rss_examples.png', dpi=70)
