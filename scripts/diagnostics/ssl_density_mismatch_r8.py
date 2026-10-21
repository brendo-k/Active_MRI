"""Why does SSDU-style test-time adaptation of the LOUPE VarNet lose PSNR? One slice, scored against the ground truth
at two input densities while the VarNet is fine-tuned on the acquired data alone (R=8).

Omega is the slice's acquired mask (seeded LOUPE draw); a fixed split gives Lambda (`--rho` of Omega's non-centre
points) and Theta = Omega - Lambda, the sparser input SSDU trains at. Each step re-splits Omega and minimises the
coil-space SSDU loss on Lambda (sense net frozen). Reported at each step: trainer-path PSNR with Omega as input
(what is deployed) and with the fixed Theta as input (what is trained), the least-squares intensity scale of each,
and the relative L2 error of the image re-projected through the sense maps, fft(x * S), on Theta (data the
VarNet was given) and on Lambda.

    python scripts/diagnostics/ssl_density_mismatch_r8.py --rho 0.4 --lr 1e-5
"""
import argparse
import copy
import json

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.image_processing import fft_2d_img
from activemri.utils.util import compute_psnr_torch
from mask_gradient_anatomy_r8 import load
from subject_coadapt_ceiling_r8 import LOUPE_RUN, PROBABILITY_EPS, Slices, masks_from_logits
from subject_ssl_tta_r8 import ssdu_step_loss


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--anatomy-index', type=int, default=0, help='0 brain, 1 knee')
    parser.add_argument('--slice', type=int, default=20, help='position in that anatomy\'s validation set')
    parser.add_argument('--rho', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--report-at', default='5,10,25,50')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def relative_l2(error, reference):
    return float(error.abs().pow(2).sum().sqrt() / reference.abs().pow(2).sum().sqrt())


@torch.no_grad()
def measure(model, part, target, omega, theta, lam):
    model.eval()
    row = {}
    for name, mask in (('omega', omega), ('theta', theta)):
        out = model(part['target'], part['kspace'], fixed_mask=mask)
        output, reference = _apply_signal_mask(out['output'], target, part['metric_mask'])
        row[f'psnr_{name}'] = float(compute_psnr_torch(output, reference))
        row[f'scale_{name}'] = float((output * reference).sum() / (output * output).sum())
        if name == 'theta':
            measured = torch.view_as_complex(part['kspace'].contiguous())
            reprojected = fft_2d_img(out['recon_complex'] * out['sense_maps'])
            row['reprojection_error_theta'] = relative_l2((reprojected - measured) * theta, measured * theta)
            row['reprojection_error_lambda'] = relative_l2((reprojected - measured) * lam, measured * lam)
    return row


def main():
    args = parse_args()
    torch.manual_seed(0)
    base, _ = load(LOUPE_RUN, 'last_model.pt', args.device)
    cfg = OmegaConf.load(LOUPE_RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 0
    _, val_loader = build_dataloaders(cfg)
    datasets = val_loader.dataset.datasets
    slices = Slices(datasets[args.anatomy_index], sum(len(d) for d in datasets[:args.anatomy_index]), val_loader.collate_fn, args.device)
    part = slices.batch([args.slice])
    target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
    with torch.no_grad():
        logits = torch.logit(base.propose_mask(part['kspace'])['unscaled'].clamp(PROBABILITY_EPS, 1 - PROBABILITY_EPS))
        torch.manual_seed(slices.global_index(args.slice))
        omega = masks_from_logits(base, logits, part['kspace'])
        centre, _ = base._initial_masks(part['kspace'])
        generator = torch.Generator(device=omega.device).manual_seed(1)
        lam = (omega - centre) * (torch.rand(omega.shape, generator=generator, device=omega.device) < args.rho)
        theta = omega - lam

    model = copy.deepcopy(base)
    rows = [{'step': 0, **measure(model, part, target, omega, theta, lam)}]
    parameters = [p for name, p in model.named_parameters() if name.startswith('reconstructor.')]
    for p in parameters:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    report_at = {int(s) for s in args.report_at.split(',')}
    for step in range(1, args.steps + 1):
        model.train()
        model.sampler.eval()
        loss_mask = (omega - centre) * (torch.rand_like(omega) < args.rho)
        optimizer.zero_grad()
        loss = ssdu_step_loss(model, part, omega - loss_mask, loss_mask, 'coil')
        loss.backward()
        optimizer.step()
        if step in report_at:
            rows.append({'step': step, 'train_loss': float(loss), **measure(model, part, target, omega, theta, lam)})
    for row in rows:
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)


if __name__ == '__main__':
    main()
