"""Probes whether a trained SequentialSampling model exploits Hermitian (conjugate) k-space symmetry.

If the coil-combined image x = S^H y is close to real, its k-space obeys
X(-k) = conj(X(k)), so a sampler can skip -k whenever it samples k and leave a
"void" that the reconstructor fills in (partial Fourier / homodyne). Tests:

    1 realness     Is x close to real? Compared across sense-map variants
                   (two normalisations, ESPIRiT).
    2 mask         Do learned masks pair up k and -k less often than chance?
    3 kspace       Is the recon's k-space error lower at unsampled k whose Hermitian
                   partner was sampled than at k whose mirror-in-one-axis "placebo"
                   partner was sampled? (Placebo partners have the same geometry but
                   no conjugate relationship.)
    4 interventions  Same reconstructor on learned / random / Hermitian-paired /
                   partner-flipped masks, next to CG-SENSE on the same masks.
    5 phase        Multiplies the object by a mid-frequency random phase (too fast for
                   the 5x5 sense estimate to absorb) and reruns tests 2-3 plus NMSE.
    6 antilinearity  Perturbs one measured sample by d and i*d. A conjugate mechanism
                   responds at -k with conj(d) (antilinear); interpolation responds with d.

Partner of centred index i (DC at N//2) is (N - i) % N, i.e. flip then roll by one.

Usage:
    python scripts/probe_conjugate_symmetry.py \
        --checkpoint checkpoints/onestep_2d_varnet_brain_knee_samp128/best_model.pt \
        --device cuda:3
"""
import argparse
import json
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import _path  # noqa: F401
import pathlib

import einops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from activemri.datasets.fastmri_data import FastMRIDataset, _collate_fastmri_batch
from activemri.reconstructors.CG_Sense import conjugate_gradient_sense
from activemri.training.trainer import anatomy_labels_from_batch, build_model
from activemri.utils.complex_conversion import complex_to_real, real_to_complex
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img

RADIUS_EDGES = (8, 16, 32, 48, 64, 96, 128, 182)
TESTS = (1, 2, 3, 4, 5, 6)
torch.set_grad_enabled(False)


def partner(t):
    """Hermitian partner k -> -k on the last two (centred) dims."""
    return torch.roll(torch.flip(t, (-2, -1)), (1, 1), (-2, -1))


def mirror_ky(t):
    return torch.roll(torch.flip(t, (-2,)), 1, -2)


def mirror_kx(t):
    return torch.roll(torch.flip(t, (-1,)), 1, -1)


def radius_bin_index(shape, device):
    h, w = shape
    ky = torch.arange(h, device=device) - h // 2
    kx = torch.arange(w, device=device) - w // 2
    radius = torch.hypot(ky[:, None].float(), kx[None, :].float())
    edges = torch.tensor(RADIUS_EDGES, device=device, dtype=radius.dtype)
    index = torch.bucketize(radius, edges, right=True) - 1
    index[(radius < edges[0]) | (radius >= edges[-1])] = -1
    return index


def binned_sum(values, region, bin_index):
    index = bin_index.expand_as(region)
    selected = region & (index >= 0)
    return torch.bincount(index[selected], weights=values[selected].double(), minlength=len(RADIUS_EDGES) - 1)


def bin_labels():
    return [f"{lo}-{hi}" for lo, hi in zip(RADIUS_EDGES[:-1], RADIUS_EDGES[1:])]


def print_table(title, columns, rows):
    print(f"\n=== {title} ===")
    widths = [max(len(c), 9) for c in columns]
    print("  ".join(c.rjust(w) for c, w in zip(columns, widths)))
    for row in rows:
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        print("  ".join(c.rjust(w) for c, w in zip(cells, widths)))


def to_complex(kspace_real):
    return torch.view_as_complex(kspace_real.contiguous())


def combine(kspace, sense_maps):
    return (ifft_2d_img(kspace) * sense_maps.conj()).sum(dim=1)


def body_region(image, fraction=0.1):
    magnitude = image.abs()
    return magnitude > fraction * magnitude.amax(dim=(-2, -1), keepdim=True)


def align_global_phase(image, body):
    theta = torch.angle((image * body).sum(dim=(-2, -1)))
    return image * torch.exp(-1j * theta)[:, None, None]


def imaginary_fraction(image, body):
    image = align_global_phase(image, body)
    body = body.float()
    return ((image.imag * body).pow(2).sum(dim=(-2, -1)) / (image.abs() * body).pow(2).sum(dim=(-2, -1)))


def raw_sense_maps(net, kspace):
    """`SensetivityModel_mc.forward` up to (but excluding) the RSS normalisation."""
    images = ifft_2d_img(net.mask_center(kspace), axes=[-1, -2])
    coils = images.shape[1]
    flat = complex_to_real(einops.rearrange(images, 'b c h w -> (b c) 1 h w'))
    flat, mean, std = net.norm(flat)
    flat = net.unnorm(net.model(flat), mean, std)
    return einops.rearrange(real_to_complex(flat), '(b c) 1 h w -> b c h w', c=coils)


def realness_variants(model, kspace, espirit_maps):
    coil_images = ifft_2d_img(kspace)
    variants = {}
    raw = raw_sense_maps(model.sensetivity_net, kspace)
    unit = raw / raw.abs().pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
    variants['rss_norm'] = (coil_images * unit.conj()).sum(dim=1)
    energy = raw.abs().pow(2).sum(dim=1).clamp(min=1e-8)
    variants['sumsq_norm'] = (coil_images * raw.conj()).sum(dim=1) / energy
    if espirit_maps is not None:
        variants['espirit'] = (coil_images * espirit_maps.conj()).sum(dim=1)
    return variants


def rss_magnitude(image, sense_maps):
    coil_images = image * sense_maps.conj()
    return coil_images.abs().pow(2).sum(dim=1, keepdim=True).sqrt()


def nmse(pred, target):
    pred, target = pred.flatten(1), target.flatten(1)
    return ((pred - target).pow(2).sum(1) / target.pow(2).sum(1).clamp(min=1e-12)).cpu()


def random_phase_field(batch, shape, low, high, amplitude, device, generator):
    h, w = shape
    noise = torch.randn(batch, h, w, device=device, generator=generator)
    spectrum = torch.fft.fft2(noise)
    fy = torch.fft.fftfreq(h, device=device) * h
    fx = torch.fft.fftfreq(w, device=device) * w
    radius = torch.hypot(fy[:, None], fx[None, :])
    band = ((radius >= low) & (radius <= high)).float()
    field = torch.fft.ifft2(spectrum * band).real
    field = field / field.std(dim=(-2, -1), keepdim=True)
    return amplitude * field


def scramble_phase(kspace, field):
    coil_images = ifft_2d_img(kspace)
    return fft_2d_img(coil_images * torch.exp(1j * field)[:, None])


class ConditionStats:
    """Accumulates tests 2, 3 and NMSE for one data condition (original or phase-scrambled)."""

    def __init__(self, shape, device):
        self.bin_index = radius_bin_index(shape, device)
        self.num_bins = len(RADIUS_EDGES) - 1
        self.masks = []
        self.nmse = []
        self.groups = ('herm', 'mirror_ky', 'mirror_kx', 'none')
        zeros = lambda: torch.zeros(self.num_bins, dtype=torch.float64, device=device)
        self.error = {g: zeros() for g in self.groups}
        self.energy = {g: zeros() for g in self.groups}
        self.count = {g: zeros() for g in self.groups}
        self.oracle_error = zeros()

    def update(self, pred_dict, kspace, target, valid_mask):
        mask = pred_dict['mask']
        sense_maps = pred_dict['sense_maps']
        self.masks.append((mask[:, 0] > 0).cpu())
        self.nmse.append(nmse(pred_dict['output'], target.unsqueeze(1)))

        recon = pred_dict['intermediate_recons'][-1][:, 0]
        truth = combine(kspace, sense_maps)
        body = body_region(truth)
        truth, recon = align_global_phase(truth, body), align_global_phase_like(recon, truth, body)
        truth_k, recon_k = fft_2d_img(truth), fft_2d_img(recon)

        error = (recon_k - truth_k).abs().pow(2)
        energy = truth_k.abs().pow(2)
        hermitian_fill_error = (torch.conj(partner(truth_k)) - truth_k).abs().pow(2)

        sampled = mask[:, 0] > 0
        valid = valid_mask[:, 0] > 0
        unsampled = ~sampled & valid & partner(valid)
        groups = {
            'herm': unsampled & partner(sampled),
            'mirror_ky': unsampled & mirror_ky(sampled),
            'mirror_kx': unsampled & mirror_kx(sampled),
            'none': unsampled & ~partner(sampled),
        }
        for name, region in groups.items():
            self.error[name] += binned_sum(error, region, self.bin_index)
            self.energy[name] += binned_sum(energy, region, self.bin_index)
            self.count[name] += binned_sum(torch.ones_like(error), region, self.bin_index)
        self.oracle_error += binned_sum(hermitian_fill_error, groups['herm'], self.bin_index)

    def mask_report(self, generator_seed=0):
        masks = torch.cat(self.masks).float()
        density = masks.mean(dim=0)
        density_partner = partner(density)
        bin_index = self.bin_index.cpu()
        region = torch.ones_like(density, dtype=torch.bool)

        def binned(values):
            return binned_sum(values[None], region[None], bin_index).numpy()

        counts = binned(torch.ones_like(density))
        both = binned((masks * partner(masks)).mean(dim=0))
        null_both = binned(density * density_partner)
        xor = binned((masks * (1 - partner(masks)) + (1 - masks) * partner(masks)).mean(dim=0))
        null_xor = binned(density * (1 - density_partner) + (1 - density) * density_partner)

        asymmetry = binned((density - density_partner).abs()) / binned(density + density_partner).clip(1e-12)
        generator = torch.Generator().manual_seed(generator_seed)
        symmetric_density = (density + density_partner) / 2
        noise_masks = (torch.rand(masks.shape, generator=generator) < symmetric_density).float()
        noise_density = noise_masks.mean(dim=0)
        noise_asymmetry = binned((noise_density - partner(noise_density)).abs()) / \
            binned(noise_density + partner(noise_density)).clip(1e-12)

        rows = [
            (label, float(counts[i]), float(both[i] / null_both[i]), float(xor[i] / null_xor[i]),
             float(asymmetry[i]), float(noise_asymmetry[i]))
            for i, label in enumerate(bin_labels())
        ]
        return rows, density.numpy()

    def kspace_report(self):
        def nmse_of(group):
            return (self.error[group] / self.energy[group].clamp(min=1e-12)).cpu().numpy()

        oracle = (self.oracle_error / self.energy['herm'].clamp(min=1e-12)).cpu().numpy()
        share = {g: (self.count[g] / sum(self.count.values()).clamp(min=1)).cpu().numpy() for g in self.groups}
        placebo = (nmse_of('mirror_ky') + nmse_of('mirror_kx')) / 2
        rows = []
        for i, label in enumerate(bin_labels()):
            herm = nmse_of('herm')[i]
            rows.append((label, float(herm), float(nmse_of('mirror_ky')[i]), float(nmse_of('mirror_kx')[i]),
                         float(nmse_of('none')[i]), float(oracle[i]), float(1 - herm / placebo[i]),
                         float(share['herm'][i])))
        return rows


def align_global_phase_like(image, reference, body):
    """Rotate `image` by the same global phase that aligns `reference`."""
    theta = torch.angle((reference * body).sum(dim=(-2, -1)))
    return image * torch.exp(-1j * theta)[:, None, None]


def mask_variants(mask, centre, valid, generator):
    free = (valid > 0) & (centre == 0)
    body = ((mask > 0) & free).float()

    count = body.flatten(1).sum(1)
    scores = torch.rand(mask.shape, device=mask.device, generator=generator).masked_fill(~free, -1)
    rank = scores.flatten(1).argsort(1, descending=True).argsort(1)
    random = (rank < count[:, None]).view_as(mask).float()

    half = body * (torch.rand(mask.shape, device=mask.device, generator=generator) < 0.5).float()
    paired = (half + partner(half)).clamp(max=1) * valid

    return {
        'learned': mask,
        'random_same_budget': (random + centre).clamp(max=1),
        'hermitian_paired': (paired + centre).clamp(max=1),
        'partner_flipped': (partner(body) * valid + centre).clamp(max=1),
    }


def intervention_test(model, kspace, target, pred_dict, valid, args, generator, results):
    sense_maps = pred_dict['sense_maps']
    centre = model._init_mask(kspace)
    recon_kwargs = {}
    variants = mask_variants(pred_dict['mask'], centre, valid, generator)
    for name, mask in variants.items():
        masked = to_complex(kspace) * mask
        varnet = model.reconstructor(masked, sense_maps, mask, **recon_kwargs)
        cg = conjugate_gradient_sense(masked, sense_maps, mask, num_iters=args.cg_iters, lamda=args.cg_lamda)
        results.setdefault(name, {'varnet': [], 'cg_sense': [], 'density': []})
        results[name]['varnet'].append(nmse(rss_magnitude(varnet, sense_maps), target.unsqueeze(1)))
        results[name]['cg_sense'].append(nmse(rss_magnitude(cg.unsqueeze(1), sense_maps), target.unsqueeze(1)))
        results[name]['density'].append(mask.flatten(1).mean(1).cpu())


def antilinearity_probe(model, kspace, pred_dict, valid, args, generator, results):
    sense_maps, mask = pred_dict['sense_maps'], pred_dict['mask']
    centre = model._init_mask(kspace)
    y = to_complex(kspace)
    h, w = mask.shape[-2:]
    ky = torch.arange(h, device=mask.device)[:, None] - h // 2
    kx = torch.arange(w, device=mask.device)[None, :] - w // 2
    radius = torch.hypot(ky.float(), kx.float())

    for b in range(min(args.probe_slices_per_batch, mask.shape[0])):
        sampled = (mask[b, 0] > 0) & (centre[b, 0] == 0)
        candidates = sampled & ~partner(mask[b, 0] > 0) & (radius > 20) & (radius < 60) & (valid[b, 0] > 0)
        positions = candidates.nonzero()
        if len(positions) == 0:
            continue
        picks = positions[torch.randperm(len(positions), device=positions.device, generator=generator)[:args.probe_locations]]
        masked = y[b:b + 1] * mask[b:b + 1]

        for iy, ix in picks.tolist():
            delta = args.probe_eps * masked[0, :, iy, ix]
            batch = masked.repeat(4, 1, 1, 1)
            for i, (sign, rotation) in enumerate([(1, 1), (-1, 1), (1, 1j), (-1, 1j)]):
                batch[i, :, iy, ix] = batch[i, :, iy, ix] + sign * rotation * delta
            recon = model.reconstructor(batch, sense_maps[b:b + 1].expand(4, -1, -1, -1),
                                        mask[b:b + 1].expand(4, -1, -1, -1))
            spectrum = fft_2d_img(recon[:, 0])
            d0 = (spectrum[0] - spectrum[1]) / 2
            d1 = (spectrum[2] - spectrum[3]) / 2
            linear = (d0 - 1j * d1) / 2
            anti = (d0 + 1j * d1) / 2

            targets = {
                'self': (iy, ix),
                'hermitian_partner': ((h - iy) % h, (w - ix) % w),
                'placebo_ky': ((h - iy) % h, ix),
                'placebo_kx': (iy, (w - ix) % w),
            }
            for name, (ty, tx) in targets.items():
                lin_e, anti_e = linear[ty, tx].abs().item() ** 2, anti[ty, tx].abs().item() ** 2
                results.setdefault(name, {'anti_fraction': [], 'response': []})
                results[name]['anti_fraction'].append(anti_e / max(lin_e + anti_e, 1e-20))
                results[name]['response'].append(d0[ty, tx].abs().item() / max(d0[iy, ix].abs().item(), 1e-20))


def build_loader(args, options):
    root = pathlib.Path(args.data_dir) / args.split
    dataset = FastMRIDataset(
        root=root, anatomy='brain', image_shape=list(options.resolution),
        num_coils=args.num_coils, num_volumes=args.num_volumes,
    )
    dataset.examples = dataset.examples[::args.stride][:args.max_slices]
    print(f"{len(dataset)} slices from {len({e[0] for e in dataset.examples})} volumes in {root}")
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=_collate_fastmri_batch,
    )


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    options = checkpoint['options']
    options.device = device
    model = build_model(options).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Loaded epoch {checkpoint['epoch']}, num_step={options.num_step}, sampler={options.sampler_type}")
    return model, options


def espirit_batch(batch, device):
    maps = batch['espirit_maps']
    if any(m is None for m in maps):
        return None
    return torch.stack(maps).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default='checkpoints/onestep_2d_varnet_brain_knee_samp128/best_model.pt')
    parser.add_argument('--data-dir', default='/home/bkadota/data/fastmri/multicoil_brain/T2')
    parser.add_argument('--split', default='val')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tests', type=int, nargs='+', default=list(TESTS), choices=TESTS)
    parser.add_argument('--num-volumes', type=int, default=4)
    parser.add_argument('--stride', type=int, default=3, help='Keep every Nth slice of the selected volumes.')
    parser.add_argument('--max-slices', type=int, default=64)
    parser.add_argument('--num-coils', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cg-iters', type=int, default=10)
    parser.add_argument('--cg-lamda', type=float, default=0.01)
    parser.add_argument('--phase-amplitude', type=float, default=1.0, help='Std of the scramble phase, radians.')
    parser.add_argument('--phase-band', type=float, nargs=2, default=(4.0, 10.0), help='Cycles per FOV.')
    parser.add_argument('--probe-eps', type=float, default=0.05)
    parser.add_argument('--probe-locations', type=int, default=6)
    parser.add_argument('--probe-slices-per-batch', type=int, default=1)
    parser.add_argument('--probe-batches', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', default='probe_conjugate_symmetry_out')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, options = load_model(args.checkpoint, device)
    if options.num_step != 1 and set(args.tests) & {4, 6}:
        raise SystemExit("Tests 4 and 6 call the reconstructor on the final mask directly and assume num_step=1.")
    loader = build_loader(args, options)
    shape = tuple(options.resolution)

    original = ConditionStats(shape, device)
    scrambled = ConditionStats(shape, device)
    realness = {}
    interventions, probes = {}, {}
    summary = {}

    for batch_idx, batch in enumerate(loader):
        kspace = batch['kspace'].to(device)
        target = batch['target'].to(device)
        labels = anatomy_labels_from_batch(batch, options.label_classes, device, options.label_key)
        valid = model._valid_kspace_mask(kspace)
        pred_dict = model(target, kspace, anatomy_labels=labels)

        if 1 in args.tests:
            for name, image in realness_variants(model, to_complex(kspace), espirit_batch(batch, device)).items():
                realness.setdefault(name, []).append(imaginary_fraction(image, body_region(image)).cpu())

        if {2, 3}.intersection(args.tests) or 5 in args.tests:
            original.update(pred_dict, to_complex(kspace), target, valid)

        if 4 in args.tests:
            intervention_test(model, kspace, target, pred_dict, valid, args, generator, interventions)

        if 5 in args.tests:
            field = random_phase_field(kspace.shape[0], shape, *args.phase_band, args.phase_amplitude, device, generator)
            scrambled_k = scramble_phase(to_complex(kspace), field)
            scrambled_real = torch.view_as_real(scrambled_k)
            scrambled_dict = model(target, scrambled_real, anatomy_labels=labels)
            scrambled.update(scrambled_dict, scrambled_k, target, valid)

        if 6 in args.tests and batch_idx < args.probe_batches:
            antilinearity_probe(model, kspace, pred_dict, valid, args, generator, probes)

        print(f"batch {batch_idx + 1}/{len(loader)}", end='\r')
    print()

    if 1 in args.tests:
        rows = [(name, float(torch.cat(v).mean()), float(torch.cat(v).median())) for name, v in realness.items()]
        print_table("1 realness: Im^2/|x|^2 after global phase alignment (lower = more real)",
                    ['variant', 'mean', 'median'], rows)
        summary['realness'] = {r[0]: {'mean': r[1], 'median': r[2]} for r in rows}

    if 2 in args.tests:
        rows, density = original.mask_report(args.seed)
        print_table("2 mask pairing (radius px). both/xor are measured / null from the mean mask; "
                    "asym vs noise is |pi(k)-pi(-k)| / (pi(k)+pi(-k)) and its finite-sample floor",
                    ['radius', 'pixels', 'both/null', 'xor/null', 'asym', 'asym_noise'], rows)
        summary['mask_pairing'] = rows
        save_mask_figure(density, out_dir / 'mean_mask_asymmetry.png')

    if 3 in args.tests:
        rows = original.kspace_report()
        print_table("3 k-space NMSE at unsampled k, by where its partner lies. herm = Hermitian partner sampled, "
                    "mirror_* = placebo, none = partner unsampled, oracle = conj fill using GT partner. "
                    "gap = 1 - herm / mean(placebo)",
                    ['radius', 'herm', 'mirror_ky', 'mirror_kx', 'none', 'oracle', 'gap', 'share_herm'], rows)
        summary['kspace_error'] = rows

    if 4 in args.tests:
        rows = []
        for name, r in interventions.items():
            varnet, cg = float(torch.cat(r['varnet']).mean()), float(torch.cat(r['cg_sense']).mean())
            rows.append((name, float(torch.cat(r['density']).mean()), varnet, cg, varnet / cg))
        print_table("4 interventions: image NMSE (same trained reconstructor and sense maps)",
                    ['mask', 'density', 'varnet', 'cg_sense', 'varnet/cg'], rows)
        summary['interventions'] = rows

    if 5 in args.tests:
        print(f"\n=== 5 phase scramble ({args.phase_band[0]}-{args.phase_band[1]} cyc/FOV, "
              f"{args.phase_amplitude} rad) ===")
        print(f"NMSE original {float(torch.cat(original.nmse).mean()):.5f}  "
              f"scrambled {float(torch.cat(scrambled.nmse).mean()):.5f}")
        rows = scrambled.kspace_report()
        print_table("5 k-space NMSE on scrambled data (compare the gap column to test 3)",
                    ['radius', 'herm', 'mirror_ky', 'mirror_kx', 'none', 'oracle', 'gap', 'share_herm'], rows)
        mask_rows, _ = scrambled.mask_report(args.seed)
        print_table("5 mask pairing on scrambled data",
                    ['radius', 'pixels', 'both/null', 'xor/null', 'asym', 'asym_noise'], mask_rows)
        summary['phase_scramble'] = {
            'nmse_original': float(torch.cat(original.nmse).mean()),
            'nmse_scrambled': float(torch.cat(scrambled.nmse).mean()),
            'kspace_error': rows,
        }

    if 6 in args.tests:
        rows = [
            (name, float(np.median(r['anti_fraction'])), float(np.median(r['response'])), len(r['anti_fraction']))
            for name, r in probes.items()
        ]
        print_table("6 antilinearity: anti/(lin+anti) of the response to a perturbation at k "
                    "(1 = pure conj, 0 = pure linear); resp = |response| / |response at k|",
                    ['location', 'anti_frac', 'resp', 'n'], rows)
        summary['antilinearity'] = rows

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWrote {out_dir}/summary.json")


def save_mask_figure(density, path):
    density = torch.from_numpy(density)
    difference = density - partner(density)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, image, title, cmap in [
        (axes[0], density, 'Mean mask pi(k)', 'viridis'),
        (axes[1], partner(density), 'Hermitian partner pi(-k)', 'viridis'),
        (axes[2], difference, 'pi(k) - pi(-k)', 'RdBu'),
    ]:
        shown = ax.imshow(image.numpy(), cmap=cmap)
        ax.set_title(title)
        fig.colorbar(shown, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == '__main__':
    main()
