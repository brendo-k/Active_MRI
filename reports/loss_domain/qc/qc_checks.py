"""Three checks on the RSS-L1 control checkpoint: (1) is the image-space mask M a no-op on the prediction (recon_complex
and S * recon_complex outside M)? (2) does the magnitude-matched target target * x/|x| remove the mismatch?
(3) PDFS SSIM by field strength."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
REPO = Path('/home/bkadota/code/active_mri/SeqMRI'); sys.path.insert(0, str(REPO))
from activemri.datasets.complex_image_cache import ComplexImageCache, cache_signature
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import build_model
from activemri.utils.metric_mask import metric_support_mask
from activemri.utils.metrics import masked_ssim

def masked_psnr(p, t, m):
    mse = ((t - p) ** 2 * m).sum(dim=(-2, -1)) / m.sum(dim=(-2, -1)).clamp(min=1.0)
    return 10 * torch.log10(t.amax(dim=(-2, -1)) ** 2 / mse)

run = Path(sys.argv[1]); out = Path(sys.argv[2])
field = {r['file']: r['field'] for r in csv.DictReader(open(REPO / 'configs/datasets/fastmri_local_v1/manifest.csv'))}
ckpt = torch.load(run / 'last_model.pt', map_location='cpu', weights_only=False)
model = build_model(ckpt['options']).cuda(); model.load_state_dict(ckpt['model']); model.eval()
_, val_loader = build_dataloaders(OmegaConf.load(run / '.hydra' / 'config.yaml'))
cache = ComplexImageCache('/export/scratch/bkadota/fastmri_complex_espirit')
torch.manual_seed(1234)
rows = defaultdict(list)
with torch.no_grad():
    for batch in val_loader:
        kr = batch['kspace'].cuda(); k = torch.view_as_complex(kr.contiguous()); t = batch['target'].cuda()
        an = list(batch['anatomy'])
        o = model(t, kr)
        xh = o['recon_complex'][:, 0]; S = o['sense_maps']
        x = torch.stack([cache.load(a, f, int(s), cache_signature(a, [320, 320], 20, 0.2)) for a, f, s in zip(an, batch['fname'], batch['slice_id'])]).cuda()
        M = (x.abs() > 0).float()
        m = metric_support_mask(k, an)[:, 0]
        xt = t * torch.where(x.abs() > 0, x / x.abs().clamp(min=1e-30), torch.zeros_like(x))
        out_rss = o['output'][:, 0].float()
        rows['condition'] += list(batch['condition']); rows['field'] += [field[f] for f in batch['fname']]
        rows['recon_outside_M_l1_frac'] += ((1 - M) * xh.abs()).sum(dim=(-2, -1)).div(xh.abs().sum(dim=(-2, -1))).tolist()
        rows['recon_outside_M_max_ratio'] += ((1 - M) * xh.abs()).amax(dim=(-2, -1)).div(xh.abs().amax(dim=(-2, -1))).tolist()
        sx = S * xh[:, None]
        rows['Simage_outside_M_l1_frac'] += ((1 - M)[:, None] * sx.abs()).sum(dim=(-3, -2, -1)).div(sx.abs().sum(dim=(-3, -2, -1))).tolist()
        rows['xtilde_vs_target_maxabs_on_metric'] += ((xt.abs() - t).abs() * m).amax(dim=(-2, -1)).tolist()
        rows['psnr_masked'] += masked_psnr(out_rss, t, m).tolist()
        rows['psnr_pseudo_xtilde'] += masked_psnr(xt.abs() + (out_rss - t), t, m).tolist()
        rows['ssim_masked'] += masked_ssim(out_rss[:, None], t[:, None], m[:, None]).float().tolist()
summary = {}
groups = sorted({(c, f) for c, f in zip(rows['condition'], rows['field'])}) + [(c, 'all') for c in sorted(set(rows['condition']))] + [('all', 'all')]
for c, f in groups:
    keep = [(c in ('all', rc)) and (f in ('all', rf)) for rc, rf in zip(rows['condition'], rows['field'])]
    summary[f'{c}/{f}'] = {'n': sum(keep), **{key: float(np.mean([v for v, kk in zip(rows[key], keep) if kk])) for key in rows if key not in ('condition', 'field')}}
    summary[f'{c}/{f}']['recon_outside_M_max_ratio_max'] = float(np.max([v for v, kk in zip(rows['recon_outside_M_max_ratio'], keep) if kk]))
out.write_text(json.dumps(summary, indent=2))
for g, s in summary.items():
    print(g, ' '.join(f'{k}={v:.4g}' for k, v in s.items()))
