"""Phase of the complex images in the loss_domain round 2, on the median-PSNR val slice of each condition (rss arm's
per-slice PSNR from the eval cache; eval mode, mask draw 1234).

Columns: the RSS target; x_E, the cached ESPIRiT image (BART ecalib, the complex arm's target); the same ESPIRiT image
recomputed with `-P` (no rotation to the coils' first principal component); x_L = sum_c conj(S_c) F^-1(k_c), the fully
sampled data through the rss arm's learned maps; the rss and complex arms' reconstructions x̂; and complex's x̂ against
x_E. Phase is shown inside the metric mask. Per slice it prints tissue medians of |phase| and of the phase differences,
and checks the cache against a fresh BART run.

usage: python reports/loss_domain/diag/phase_maps.py --out PNG [--runs-root DIR] [--eval-cache DIR]
"""
import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
sys.path.insert(0, 'scripts/eval'); sys.path.insert(0, '.')
import eval_loss_domain as e
from activemri.datasets.fastmri_data import build_manifest_dataloaders
from activemri.training.trainer import build_model
from activemri.utils.bart_ecalib import PRESETS, ecalib_sense_maps
from activemri.utils.image_processing import ifft_2d_img
from activemri.utils.metric_mask import metric_support_mask

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True, type=Path)
parser.add_argument('--runs-root', type=Path, default=Path.home() / 'data/seqmri_checkpoints/loss_domain/runs_v2')
parser.add_argument('--eval-cache', type=Path,
                    default=Path.home() / 'data/seqmri_checkpoints/loss_domain/report_untracked/eval_v2/runs')
args = parser.parse_args()
device = torch.device('cuda')
CONDITIONS = ('brain', 'knee-PD', 'knee-PDFS')

scores = torch.load(args.eval_cache / f"{e.run_name('rss', 42)}.pt", weights_only=False)
chosen = []
for condition in CONDITIONS:
    index = [i for i, c in enumerate(scores['condition']) if c == condition]
    psnr = scores['psnr'][index]
    chosen.append(index[int(torch.argsort(psnr)[len(index) // 2])])

_, val_loader = build_manifest_dataloaders(manifest='configs/datasets/fastmri_local_v1/manifest.csv', roots=e.ROOTS,
    classes=['FLAIR', 'knee'], image_shape=[320, 320], batch_size=16, num_coils=20, num_workers=8, complex_cache_dir=e.CACHE)
batch = val_loader.collate_fn([val_loader.dataset[i] for i in chosen])
for i, slot in enumerate(chosen):
    assert batch['condition'][i] == scores['condition'][slot], 'eval cache order != val dataset order'
    assert f"{batch['anatomy'][i]}/{batch['fname'][i]}" == scores['volume'][slot], 'eval cache order != val dataset order'

kr = batch['kspace'].to(device); t = batch['target'].to(device)
k = torch.view_as_complex(kr.contiguous())
coils = ifft_2d_img(k)
xe = torch.view_as_complex(batch['complex_target'].to(device).contiguous())
metric = metric_support_mask(k, list(batch['anatomy']))[:, 0].bool()
recon, maps = {}, {}
for arm in ('rss', 'complex'):
    ck = torch.load(args.runs_root / e.run_name(arm, 42) / 'last_model.pt', map_location='cpu', weights_only=False)
    model = build_model(ck['options']).to(device)
    model.load_state_dict({key.removeprefix('module.'): v for key, v in ck['model'].items()}); model.eval()
    model.sampler.eval_generator = None
    torch.manual_seed(1234)
    with torch.no_grad():
        pred = model(t, kr)
    recon[arm], maps[arm] = pred['recon_complex'][:, 0], pred['sense_maps']
    del model; torch.cuda.empty_cache()
xl = (coils * maps['rss'].conj()).sum(dim=1)

espirit_default, espirit_p = [], []
for i, anatomy in enumerate(batch['anatomy']):
    for out, extra in ((espirit_default, ()), (espirit_p, ('-P',))):
        S, _ = ecalib_sense_maps(k[i].cpu(), **PRESETS[anatomy], extra=extra)
        S = S.to(device)
        out.append(((coils[i] * S.conj()).sum(0), S))


def median_abs(phase, mask):
    return float(phase[mask].abs().median())


def relative(a, b, mask):
    return float(((a - b).abs().pow(2) * mask).sum() / (b.abs().pow(2) * mask).sum()) ** 0.5


columns = [
    ('RSS target t', None),
    ('∠x_E: cached ESPIRiT\n(complex arm target)', lambda i: torch.angle(xe[i])),
    ('∠ ESPIRiT with -P\n(no PC rotation)', lambda i: torch.angle(espirit_p[i][0])),
    ('∠x_L: full data through\nlearned maps (rss arm)', lambda i: torch.angle(xl[i])),
    ('∠x̂ rss arm\n(reconstruction)', lambda i: torch.angle(recon['rss'][i])),
    ('∠x̂ complex arm\n(reconstruction)', lambda i: torch.angle(recon['complex'][i])),
    ('∠(x̂ complex · conj x_E)\n(complex arm error)', lambda i: torch.angle(recon['complex'][i] * xe[i].conj())),
]
fig, axes = plt.subplots(len(chosen), len(columns), figsize=(2.6 * len(columns), 2.9 * len(chosen)), squeeze=False)
for i in range(len(chosen)):
    shown = metric[i].cpu().numpy()
    for j, (title, phase) in enumerate(columns):
        ax = axes[i, j]
        if phase is None:
            ax.imshow(t[i].cpu().numpy(), cmap='gray')
        else:
            ax.set_facecolor('0.85')
            image = ax.imshow(np.where(shown, phase(i).cpu().numpy(), np.nan), cmap='twilight', vmin=-np.pi, vmax=np.pi)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(title, fontsize=9)
    axes[i, 0].set_ylabel(f"{batch['condition'][i]}\n{batch['fname'][i]} s{int(batch['slice_id'][i])}", fontsize=8)
bar = fig.colorbar(image, ax=axes, fraction=0.012, pad=0.01, ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
bar.ax.set_yticklabels(['−π', '−π/2', '0', 'π/2', 'π']); bar.set_label('phase (rad)')
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=110, bbox_inches='tight')

for i in range(len(chosen)):
    tissue = metric[i] & (xe[i].abs() > 0.3 * xe[i].abs().max())
    w = tissue.float()
    xd, Sd = espirit_default[i]
    xp, Sp = espirit_p[i]
    print(f"== {batch['condition'][i]} {batch['fname'][i]} slice {int(batch['slice_id'][i])} "
          f"(rss PSNR_Masked {float(scores['psnr'][chosen[i]]):.2f}, tissue {int(w.sum())} px)")
    print(f"   cache x_E vs fresh BART default: rel err {relative(xe[i], xd, w):.4f}")
    print(f"   coil fit |S x - coils| / |coils| in tissue: ESPIRiT default {relative(Sd * xd, coils[i], w):.3f}, "
          f"-P {relative(Sp * xp, coils[i], w):.3f}, learned (rss) {relative(maps['rss'][i] * xl[i], coils[i], w):.3f}")
    for name, image in (('x_E (cache)', xe[i]), ('ESPIRiT -P', xp), ('x_L', xl[i]), ('x̂ rss', recon['rss'][i]),
                        ('x̂ complex', recon['complex'][i])):
        print(f"   median |phase| in tissue  {name:12s} {median_abs(torch.angle(image), tissue):.3f}")
    for name, a, b in (('x̂ rss vs x_L', recon['rss'][i], xl[i]), ('x_L vs x_E', xl[i], xe[i]),
                       ('x̂ complex vs x_E', recon['complex'][i], xe[i]), ('x̂ complex vs x_L', recon['complex'][i], xl[i]),
                       ('-P vs x_E', xp, xe[i])):
        print(f"   median |phase difference| {name:18s} {median_abs(torch.angle(a * b.conj()), tissue):.3f}")
print(f'figure: {args.out}')
