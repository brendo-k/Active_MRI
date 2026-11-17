"""Sampling density (9x9 box-smoothed final mask, draw seed 1234) on the lab-meeting slide's three knee slices, for kscr2's
one-step LOUPE and 16-step adaptive arms and kscr3's one-step LOUPE and SeqMRI arms (current checkpoints), on DEVICE.
Usage: python render_mask_compare.py <out.png>
"""
import shutil
import sys
import tempfile
from pathlib import Path

WORKTREE = Path('/home/bkadota/code/active_mri/SeqMRI-worktrees/exp-multistep_adaptivity')
sys.path.insert(0, str(WORKTREE / 'scripts' / 'eval'))
sys.path.insert(0, str(WORKTREE))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402

import eval_multistep_adaptivity as ev  # noqa: E402
from activemri.training.builders import build_dataloaders  # noqa: E402
from activemri.training.trainer import build_model  # noqa: E402
from activemri.utils.multistep import step_thresholds  # noqa: E402

DATA = Path.home() / 'data/seqmri_checkpoints'
SCORES = WORKTREE / 'reports/multistep_adaptivity/kscr2/scores'
ROWS = {'Knee PD': 314, 'Knee PDFS, patient A': 115, 'Knee PDFS, patient B': 550}  # matched from the slide's notes
ARMS = [('kscr2: 1-step LOUPE\nper contrast, 10x10 centre', DATA / 'multistep_kscr2/T_1_s42_kscr2'),
        ('kscr2: adaptive, 16 steps\n10x10 centre', DATA / 'multistep_kscr2/A_16_s42_kscr2'),
        ('kscr3: 1-step LOUPE (L)\none map, 40x40 centre', DATA / 'multistep_kscr3/L_1_s42_kscr3'),
        ('kscr3: SeqMRI arm (F), 4 steps\n40x40 centre', DATA / 'multistep_kscr3/F_4_s42_kscr3')]
DRAW_SEED = 1234
SMOOTH = 9
DEVICE = 'cuda'


def load_run_copy(run):
    """(model, options, epoch) of a run's last_model.pt, copied first because training may be rewriting it."""
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / 'last_model.pt'
        shutil.copy(run / 'last_model.pt', copy)
        checkpoint = torch.load(copy, map_location=DEVICE, weights_only=False)
    options = checkpoint['options']
    options.compile_unets = False
    model = build_model(options)
    model.load_state_dict(checkpoint['model'])
    return model.to(DEVICE).eval().requires_grad_(False), options, int(checkpoint['epoch'])


def val_set():
    cfg = OmegaConf.load(DATA / 'multistep_kscr3/F_4_s42_kscr3/.hydra/config.yaml')
    cfg.datasets.num_workers = 0
    _, val_loader = build_dataloaders(cfg)
    return val_loader.dataset, val_loader.collate_fn


@torch.no_grad()
def main(out):
    torch.set_num_threads(8)
    reference = np.load(SCORES / 'T_1_s42_kscr2.npz', allow_pickle=True)
    dataset, collate = val_set()
    indices = {label: int(reference['dataset_index'][row]) for label, row in ROWS.items()}
    for label, row in ROWS.items():
        fname, slice_id = dataset.examples[indices[label]] if hasattr(dataset, 'examples') else (None, None)
        if fname is not None and (fname.name, slice_id) != (reference['volume'][row], int(reference['slice_id'][row])):
            raise SystemExit(f'{label}: dataset index {indices[label]} is {fname.name}:{slice_id}, the scores say '
                             f"{reference['volume'][row]}:{reference['slice_id'][row]}")
    densities, titles = {}, []
    for title, run in ARMS:
        model, options, epoch = load_run_copy(run)
        titles.append(f'{title}, epoch {epoch + 1}')
        for label, index in indices.items():
            part = ev.labelled(collate([dataset[index]]), options, DEVICE)
            thresholds = step_thresholds(DRAW_SEED, [index], model.num_step, model.shape, DEVICE)
            final = model.sample_trajectory(part['kspace'], part['labels'], thresholds)[0][-1][0, 0].float().cpu().numpy()
            densities[title, label] = uniform_filter(final, size=SMOOTH)
            print(f'{run.name} {label}: {final.mean():.4f} sampled', flush=True)
        del model
        torch.cuda.empty_cache()
    fig, axes = plt.subplots(len(ROWS), len(ARMS), figsize=(4.2 * len(ARMS), 4.2 * len(ROWS) + 0.8))
    for r, label in enumerate(ROWS):
        for c, (title, _) in enumerate(ARMS):
            ax = axes[r, c]
            ax.imshow(densities[title, label], cmap='magma', vmin=0, vmax=0.5, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=12)
            if c == 0:
                ax.set_ylabel(label, fontsize=13)
    fig.suptitle(f'Sampling density at R=8 ({SMOOTH}x{SMOOTH} box-smoothed final mask, draw seed {DRAW_SEED}), '
                 'the lab-meeting slide\'s three knee validation slices; colour scale 0 to 0.5', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=110)
    print(f'saved {out}')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
