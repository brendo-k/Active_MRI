"""How real is the coil-combined target t = sum_c conj(S_c) IFFT(y_c) through each model's own sensitivity maps,
and how much of its spectrum is conjugate-symmetric? Repeated with coil 0's phase put back into t, which stands in
for a map set that keeps the object phase.

real share      = energy of Re(t e^{-i phi0}) / energy of t inside the metric mask, phi0 the best global phase.
hermitian share = energy of the conjugate-symmetric part of FFT(t * metric mask) / total energy.
Both are fractions in [0, 1]; a real image gives 1 for both."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / 'scripts'), str(REPO / 'scripts' / 'eval')]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_multistep_adaptivity as ema  # noqa: E402

ARMS = ('Lrss_1_s42_tF', 'Lrss_1_s43_tF', 'Lcplx_1_s42_tF', 'Lcplx_1_s43_tF')


def centred_ifft2(kspace):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1))), dim=(-2, -1))


def real_share(image, mask):
    values = image[mask]
    best_phase = 0.5 * torch.angle((values * values).sum())
    rotated = values * torch.exp(-1j * best_phase)
    return float((rotated.real ** 2).sum() / (values.abs() ** 2).sum())


def hermitian_share(image, mask):
    spectrum = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image * mask)))
    mirrored = torch.roll(torch.flip(spectrum, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1)).conj()
    hermitian = 0.5 * (spectrum + mirrored)
    return float((hermitian.abs() ** 2).sum() / (spectrum.abs() ** 2).sum())


@torch.no_grad()
def score_run(run, data_root, n_slices, device):
    model, checkpoint = ema.load_run(run, 'last_model.pt', device)
    options = checkpoint['options']
    cfg = ema.datasets_config(run, SimpleNamespace(workers=0, val_half='dev', data_root=[data_root]))
    cfg.datasets.train_tier = 'S'
    _, loader = ema.build_dataloaders(cfg)
    dataset, collate = loader.dataset, loader.collate_fn
    rows = {'own_maps_real': [], 'own_maps_hermitian': [], 'coil0_phase_real': [], 'coil0_phase_hermitian': []}
    for index in np.linspace(0, len(dataset) - 1, n_slices).astype(int):
        part = ema.labelled(collate([dataset[int(index)]]), options, device)
        maps, _ = model.sense_maps_for(part['kspace'])
        coils = centred_ifft2(torch.view_as_complex(part['kspace'].contiguous()))
        combined = (coils * maps.conj()).sum(dim=1)[0]
        mask = part['metric_mask'][0, 0] > 0
        with_object_phase = combined * torch.exp(1j * torch.angle(maps[0, 0]))
        rows['own_maps_real'].append(real_share(combined, mask))
        rows['own_maps_hermitian'].append(hermitian_share(combined, mask))
        rows['coil0_phase_real'].append(real_share(with_object_phase, mask))
        rows['coil0_phase_hermitian'].append(hermitian_share(with_object_phase, mask))
    del model
    torch.cuda.empty_cache()
    return {key: {'median': float(np.median(v)), 'min': float(np.min(v)), 'max': float(np.max(v))}
            for key, v in rows.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', type=Path, required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--slices', type=int, default=24)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = {'slices_per_run': args.slices, 'split': 'FLAIR dev half, evenly spaced slices', 'runs': {}}
    for name in ARMS:
        result['runs'][name] = score_run(args.runs_dir / name, args.data_root, args.slices, 'cuda')
        print(name, {k: round(v['median'], 3) for k, v in result['runs'][name].items()}, flush=True)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
