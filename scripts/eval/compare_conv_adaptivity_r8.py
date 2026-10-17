"""How much the 1-step conv_2d sampler's mask varies across slices and anatomies, per checkpoint (R=8, brain+knee).

For each run the effective sampling probability is computed per validation slice (as in
compare_conv_loupe_masks_r8.py). Adaptivity is read from:

  across_slice_std      mean over k-space of the std of the probability across slices of one anatomy
  slice_vs_mean_corr    mean correlation of each slice's map with its anatomy's mean map (1 = no per-slice adaptation)
  brain_vs_knee_corr    correlation of the brain-mean and knee-mean maps (1 = no per-anatomy adaptation)
  brain_knee_max_abs    largest |brain-mean - knee-mean| probability difference

    python scripts/eval/compare_conv_adaptivity_r8.py
"""
import argparse
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from compare_conv_loupe_masks_r8 import ANATOMIES, collect, correlation, load_model, roughness

RUNS = {
    'baseline_epoch4': (Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05'), 'resume_from_epoch4.pt'),
    'baseline_final': (Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05'), 'last_model.pt'),
    'image_input': (Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_imginput'), 'last_model.pt'),
    'draws2': (Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_draws2'), 'last_model.pt'),
    'image_input_bias': (Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_imginput_headbias'), 'last_model.pt'),
    'clamp10_leak0.05': (Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_clamp10_leak0.05'), 'last_model.pt'),
    'image_input_mlp_head': (Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_imginput_headmlp_ln_clipsplit'), 'last_model.pt'),
}
OUT_DIR = Path('reports/conv_vs_loupe_r8')


def adaptivity(maps):
    means = {a: maps[a].mean(0) for a in ANATOMIES}
    stats = {a: {'across_slice_std': float(maps[a].std(0).mean()),
                 'slice_vs_mean_corr': float(np.mean([correlation(s, means[a]) for s in maps[a]])),
                 'roughness': float(roughness(maps[a]).mean())} for a in ANATOMIES}
    stats['brain_vs_knee_corr'] = correlation(means['brain'], means['knee'])
    stats['brain_knee_max_abs'] = float(np.abs(means['brain'] - means['knee']).max())
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, default=OUT_DIR)
    parser.add_argument('--runs', nargs='+', choices=list(RUNS), default=list(RUNS),
                        help='evaluate only these runs, keeping the others already in adaptivity.json')
    args = parser.parse_args()

    cfg = OmegaConf.load(RUNS['baseline_final'][0] / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)

    out_file = args.out / 'adaptivity.json'
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    for name in args.runs:
        run_dir, checkpoint = RUNS[name]
        model, epoch = load_model(run_dir, checkpoint, args.device)
        maps = collect({'conv': model}, val_loader, args.device)['conv']
        results[name] = {'epochs_trained': epoch + 1, **adaptivity(maps)}
        print(name, json.dumps(results[name], indent=2), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({name: results[name] for name in RUNS if name in results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
