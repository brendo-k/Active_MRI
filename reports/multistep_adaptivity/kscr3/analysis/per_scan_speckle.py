"""Per-pixel sampling probability, unsmoothed, on the lab-meeting slide's three knee scans: the chance each location is in
the final mask, estimated from 256 draws (16 batches of 16 draw seeds). A checkerboard-like interleaving shows as side
neighbours that differ while diagonal neighbours agree; the annotation gives both correlations inside the zoom.
Usage: python per_scan_speckle.py <out.png>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import rotate_probe as rp  # noqa: E402
from activemri.training.builders import build_dataloaders  # noqa: E402
from activemri.utils.multistep import step_thresholds  # noqa: E402

ARMS = [('kscr2 1-step LOUPE', rp.DATA / 'multistep_kscr2/T_1_s42_kscr2'),
        ('kscr2 adaptive, 16 steps', rp.DATA / 'multistep_kscr2/A_16_s42_kscr2'),
        ('kscr3 1-step LOUPE (L)', rp.DATA / 'multistep_kscr3/L_1_s42_kscr3'),
        ('kscr3 SeqMRI arm (F)', rp.DATA / 'multistep_kscr3/F_4_s42_kscr3')]
CHUNKS, CHUNK = 16, 16
CROP = (slice(110, 190), slice(60, 140))


def inclusion_probability(model, kspace, labels, index):
    batch, total = kspace.expand(CHUNK, *kspace.shape[1:]).contiguous(), 0.0
    for chunk in range(CHUNKS):
        seeds = [step_thresholds(1000 + chunk * CHUNK + k, [index], model.num_step, model.shape, rp.DEVICE)
                 for k in range(CHUNK)]
        thresholds = [torch.cat([s[step] for s in seeds]) for step in range(model.num_step)]
        total = total + model.sample_trajectory(batch, labels.expand(CHUNK), thresholds)[0][-1].mean(dim=0)[0]
    return (total / CHUNKS).float().cpu().numpy()


def neighbour_correlations(p):
    side = np.corrcoef(p[:, :-1].ravel(), p[:, 1:].ravel())[0, 1]
    diagonal = np.corrcoef(p[:-1, :-1].ravel(), p[1:, 1:].ravel())[0, 1]
    return side, diagonal


@torch.no_grad()
def main(out):
    reference = np.load(rp.SCORES / 'T_1_s42_kscr2.npz', allow_pickle=True)
    cfg = OmegaConf.load(rp.DATA / 'multistep_kscr3/F_4_s42_kscr3/.hydra/config.yaml')
    cfg.datasets.num_workers = 0
    _, val_loader = build_dataloaders(cfg)
    dataset, collate = val_loader.dataset, val_loader.collate_fn
    scans = list(rp.ROWS)
    fig, axes = plt.subplots(len(ARMS), len(scans) + 1, figsize=(4.2 * (len(scans) + 1), 4.4 * len(ARMS) + 0.8))
    for r, (name, run) in enumerate(ARMS):
        model, options, epoch = rp.load_run_copy(run)
        for c, scan in enumerate(scans):
            index = int(reference['dataset_index'][rp.ROWS[scan]])
            part = rp.ev.labelled(collate([dataset[index]]), options, rp.DEVICE)
            p = inclusion_probability(model, part['kspace'], part['labels'], index)
            if c == 0:
                axes[r, 0].imshow(p, cmap='magma', vmin=0, vmax=1, interpolation='nearest')
                axes[r, 0].add_patch(plt.Rectangle((CROP[1].start, CROP[0].start), CROP[1].stop - CROP[1].start,
                                                   CROP[0].stop - CROP[0].start, fill=False, edgecolor='cyan'))
                axes[r, 0].set_ylabel(f'{name}\nepoch {epoch + 1}', fontsize=12)
            zoom = p[CROP]
            side, diagonal = neighbour_correlations(zoom)
            axes[r, c + 1].imshow(zoom, cmap='magma', vmin=0, vmax=1, interpolation='nearest')
            axes[r, c + 1].set_xlabel(f'neighbours: side {side:+.2f}, diagonal {diagonal:+.2f}', fontsize=11)
            print(f'{name:28s} {scan:22s} side {side:+.2f} diagonal {diagonal:+.2f}', flush=True)
        del model
        torch.cuda.empty_cache()
    axes[0, 0].set_title(f'{scans[0]}, whole grid', fontsize=12)
    for c, scan in enumerate(scans):
        axes[0, c + 1].set_title(f'{scan}, zoom (cyan box)', fontsize=12)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f'Chance each location is in the final mask, from {CHUNKS * CHUNK} draws, no smoothing (colour 0 to 1)',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=110)
    print(f'saved {out}')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
