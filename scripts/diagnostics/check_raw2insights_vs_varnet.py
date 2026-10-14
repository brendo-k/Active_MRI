"""Step 0: is NV-Raw2Insights usable as an independent judge of sampling masks?

Scores the in-house VarNet and the frozen NVIDIA model on identical, non-learned
column masks (equispaced / random, centre preselected). Conditioning labels for
the NVIDIA model are chosen on a tuning subset of the validation slices and
reported on the disjoint remainder.
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import _path  # noqa: F401
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from activemri.reconstructors.SensetivityModel_mc import SensetivityModel_mc
from activemri.reconstructors.VarNet import VarNet
from activemri.reconstructors.raw2insights import Raw2InsightsRecon

ACCELERATIONS = (4, 8, 16)  # NVIDIA was conditioned on 8/16/24 only; 4x is outside its range
MASK_TYPES = ("uniform", "gaussian", "radial")
ACQ_TYPES = ("BlackBlood", "Cine", "Flow2d", "LGE", "Mapping", "Perfusion", "T1rho", "T1w", "T2w")
PRESELECT = 10


def load_varnet(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    recon_cfg = checkpoint["options"].recon
    varnet = VarNet(num_cascades=recon_cfg.num_cascades, chans=recon_cfg.chans, pools=recon_cfg.pools)
    sense_net = SensetivityModel_mc(2, 2, 8)
    state = checkpoint["model"]
    varnet.load_state_dict({k[len("reconstructor."):]: v for k, v in state.items() if k.startswith("reconstructor.")})
    sense_net.load_state_dict({k[len("sensetivity_net."):]: v for k, v in state.items() if k.startswith("sensetivity_net.")})
    return varnet.to(device).eval(), sense_net.to(device).eval()


def make_mask(kind, acceleration, batch_size, width, generator, device):
    """DC-centred `[B, 1, H=width, W=width]`-broadcastable column mask with `width/acceleration` columns."""
    budget = width // acceleration
    centre = torch.zeros(width, dtype=torch.bool)
    centre[width // 2 - PRESELECT // 2: width // 2 + PRESELECT // 2] = True
    masks = []
    for _ in range(batch_size):
        columns = centre.clone()
        remaining = budget - int(columns.sum())
        if kind == "equispaced":
            free = (~columns).nonzero().squeeze(1)
            offset = int(torch.randint(len(free), (1,), generator=generator))
            picks = free[(offset + torch.arange(remaining) * (len(free) // remaining)) % len(free)]
        elif kind == "random":
            free = (~columns).nonzero().squeeze(1)
            picks = free[torch.randperm(len(free), generator=generator)[:remaining]]
        else:
            raise ValueError(kind)
        columns[picks] = True
        masks.append(columns)
    columns = torch.stack(masks).to(device).float()  # [B, W]
    return columns[:, None, None, :].expand(-1, 1, width, -1).contiguous()


def metrics(pred_abs, target):
    """Per-slice NMSE / SSIM / PSNR, `[B]` each. Inputs `[B, 1, H, W]` real."""
    err = (pred_abs - target).pow(2).flatten(1).sum(1)
    nmse = err / target.pow(2).flatten(1).sum(1).clamp(min=1e-12)
    data_range = target.flatten(1).amax(1)
    psnr = 10 * torch.log10(data_range.pow(2) / (err / target[0].numel()).clamp(min=1e-20))
    per_slice_ssim = torch.stack([
        ssim(pred_abs[i:i + 1], target[i:i + 1], data_range=float(data_range[i])) for i in range(len(target))
    ])
    return nmse, per_slice_ssim, psnr


def evaluate(recon_fn, slices, kind, acceleration, device, seed=0):
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for full_kspace, target, sense_maps, name in slices:
        mask = make_mask(kind, acceleration, 1, full_kspace.shape[-2], generator, device)
        masked = torch.view_as_complex((mask.unsqueeze(-1) * full_kspace).contiguous())
        with torch.no_grad():
            pred = recon_fn(masked, sense_maps, mask).abs()
        nmse, s, p = metrics(pred, target)
        rows.append({"slice": name, "nmse": nmse.item(), "ssim": s.item(), "psnr": p.item()})
    return rows


def summarize(rows):
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]) / np.sqrt(len(rows))))
            for k in ("nmse", "ssim", "psnr")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--varnet-checkpoint", default="checkpoints/onestep_2d_varnet_brain_knee_samp128/best_model.pt")
    parser.add_argument("--raw2insights-repo", default=str(Path(__file__).resolve().parents[3] / "NV-Raw2insights-MRI"))
    parser.add_argument("--anatomies", nargs="+", default=["brain", "knee"])
    parser.add_argument("--num-tune", type=int, default=6)
    parser.add_argument("--num-eval", type=int, default=40)
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--acq-type", default="T2w")
    parser.add_argument("--mask-type", default="uniform")
    parser.add_argument("--out", default="/tmp/raw2insights_step0.json")
    args = parser.parse_args()
    device = torch.device("cuda")

    varnet, sense_net = load_varnet(args.varnet_checkpoint, device)
    nvidia = Raw2InsightsRecon(args.raw2insights_repo, mask_type=args.mask_type, acq_type=args.acq_type).to(device)

    _, val_loader = build_fastmri_dataloaders(
        data_root=Path("/home/bkadota/data/fastmri"), anatomies=args.anatomies, image_shape=[256, 256],
        batch_size=1, num_coils=16, num_workers=4, contrast="T2", min_energy_fraction=0.2,
    )
    total = args.num_tune + args.num_eval
    cache = Path(args.out).with_suffix(f".slices{total}.pt")
    picks = set(np.linspace(0, len(val_loader.dataset) - 1, total).astype(int).tolist())
    slices = []
    for idx, batch in enumerate(val_loader):
        if cache.exists():
            slices = torch.load(cache, map_location=device)
            break
        if idx not in picks:
            continue
        full_kspace = batch["kspace"].to(device)
        target = batch["target"].to(device)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        with torch.no_grad():
            sense_maps, _ = sense_net(torch.view_as_complex(full_kspace.contiguous()), mask_method="pre_network")
        slices.append((full_kspace, target, sense_maps, f"{batch['fname'][0]}:{int(batch['slice_id'][0])}"))
    if not cache.exists():
        torch.save(slices, cache)
    tune, held_out = slices[::2][:args.num_tune], [s for i, s in enumerate(slices) if i % 2 == 1 or i >= 2 * args.num_tune]
    print(f"{len(tune)} tuning slices, {len(held_out)} held-out slices")

    results = {"tuning": {}, "eval": {}}
    if not args.skip_sweep:
        for mask_type, acq_type in itertools.product(MASK_TYPES, ACQ_TYPES):
            nvidia.mask_type, nvidia.acq_type = mask_type, acq_type
            rows = evaluate(nvidia, tune, "equispaced", 8, device)
            results["tuning"][f"{mask_type}/{acq_type}"] = summarize(rows)
            print(f"tune {mask_type:9s} {acq_type:10s} nmse={results['tuning'][f'{mask_type}/{acq_type}']['nmse'][0]:.4f}", flush=True)
        best = min(results["tuning"], key=lambda k: results["tuning"][k]["nmse"][0])
        nvidia.mask_type, nvidia.acq_type = best.split("/")
        print("selected conditioning:", best)
        results["selected"] = best

    for kind, acceleration in itertools.product(("equispaced", "random"), ACCELERATIONS):
        for name, fn in (("zerofill", zero_filled), ("varnet", varnet_fn(varnet)), ("raw2insights", nvidia)):
            rows = evaluate(fn, held_out, kind, acceleration, device)
            results["eval"][f"{kind}/{acceleration}x/{name}"] = summarize(rows)
            m = results["eval"][f"{kind}/{acceleration}x/{name}"]
            print(f"{kind:10s} {acceleration:2d}x {name:13s} nmse={m['nmse'][0]:.4f} ssim={m['ssim'][0]:.4f} psnr={m['psnr'][0]:.2f}", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


def zero_filled(masked_kspace, sense_maps, mask):
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(masked_kspace, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)
    ).abs().pow(2).sum(dim=1, keepdim=True).sqrt()


def varnet_fn(varnet):
    def run(masked_kspace, sense_maps, mask):
        return varnet(masked_kspace, sense_maps, mask)
    return run


if __name__ == "__main__":
    main()
