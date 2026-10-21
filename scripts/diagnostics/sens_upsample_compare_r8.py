"""Compare conv vs bilinear sensitivity-U-Net upsampling in two matched CG-SENSE LOUPE runs.

Reads the per-epoch validation metrics from the two training logs and builds a montage of the training-batch
probabilistic mask (bottom-left panel of visualizations/train_epoch=N_iter=1_step0.png) at selected epochs.
"""
import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw

SEED_RUNS = {
    42: {'conv': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared',
         'bilinear': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep'},
    43: {'conv': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared_seed43',
         'bilinear': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep_seed43'},
}
RUNS = SEED_RUNS[42]
MASK_PANEL_BOX = (224, 322, 466, 534)
METRICS = re.compile(r'\[(brain|knee)\] MSE=\S+ NMSE=(\S+) PSNR=(\S+) SSIM=(\S+)')
EPOCH = re.compile(r'Epoch = \[\s*(\d+)/')


def validation_by_epoch(log: Path) -> dict[int, dict[str, dict[str, float]]]:
    """The validation metric line is logged just before its 'Epoch = [n/N]' line."""
    pending: dict[str, dict[str, float]] = {}
    by_epoch = {}
    for line in log.read_text().splitlines():
        for anatomy, nmse, psnr, ssim in METRICS.findall(line):
            pending[anatomy] = {'nmse': float(nmse), 'psnr': float(psnr), 'ssim': float(ssim)}
        match = EPOCH.search(line)
        if match and pending:
            by_epoch[int(match.group(1))] = pending
            pending = {}
    return by_epoch


def montage(epochs: list[int], out: Path) -> None:
    left, top, right, bottom = MASK_PANEL_BOX
    width, height, label = right - left, bottom - top, 16
    image = Image.new('RGB', (width * len(epochs), (height + label) * len(RUNS)), 'white')
    draw = ImageDraw.Draw(image)
    for row, (name, run) in enumerate(RUNS.items()):
        for col, epoch in enumerate(epochs):
            frame = Image.open(f'{run}/visualizations/train_epoch={epoch}_iter=1_step0.png').convert('RGB')
            image.paste(frame.crop(MASK_PANEL_BOX), (col * width, row * (height + label) + label))
            draw.text((col * width + 4, row * (height + label) + 2), f'{name} ep{epoch}', fill='black')
    image.save(out)


def main() -> None:
    global RUNS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, nargs='+', default=[1, 2, 3, 5, 10, 20, 50])
    parser.add_argument('--out-dir', type=Path, default=Path('reports/sens_upsample_r8'))
    parser.add_argument('--seed', type=int, default=42, choices=tuple(SEED_RUNS))
    args = parser.parse_args()
    RUNS = SEED_RUNS[args.seed]
    suffix = '' if args.seed == 42 else f'_seed{args.seed}'
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {name: validation_by_epoch(Path(f'{run}.log')) for name, run in RUNS.items()}
    (args.out_dir / f'val_metrics{suffix}.json').write_text(json.dumps(metrics, indent=1))
    montage(args.epochs, args.out_dir / f'mask_montage{suffix}.png')
    for epoch in args.epochs:
        for anatomy in ('brain', 'knee'):
            cells = '  '.join(f"{name}: {metrics[name][epoch][anatomy]['psnr']:.2f} dB / {metrics[name][epoch][anatomy]['ssim']:.4f}"
                              for name in RUNS)
            print(f'ep{epoch:>2} {anatomy:<5} {cells}')


if __name__ == '__main__':
    main()
