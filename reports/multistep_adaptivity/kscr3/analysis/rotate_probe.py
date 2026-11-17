"""Does a sampler's sampling density rotate when the image does?

The slide's three knee validation slices are rotated by exactly 90 degrees: the coil k-space is rotated about its DC
sample (rot90, then a roll that puts DC back), which rotates every coil image the same way with no interpolation. For
each arm, the expected final-mask density is the mean of 16 draws (draw seeds 0-15), with and without the rotation.
"Follows the image" is the correlation of the rotated input's density with the original density rotated; "stays put" is
its correlation with the original density as it was. Both are taken over the acquired k-space outside the arm's
pre-selected centre, on 9x9 box-smoothed densities. A static mask stays put exactly; a sampler that adapts to the
anatomy's orientation follows it.
Usage: python rotate_probe.py <out.png>
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
ROWS = {'Knee PD': 314, 'Knee PDFS, patient A': 115, 'Knee PDFS, patient B': 550}
FIGURE_SLICE = 'Knee PD'
ARMS = [('kscr3 SeqMRI arm (F)', DATA / 'multistep_kscr3/F_4_s42_kscr3'),
        ('kscr3 1-step LOUPE (L)', DATA / 'multistep_kscr3/L_1_s42_kscr3'),
        ('kscr2 adaptive, 16 steps (A)', DATA / 'multistep_kscr2/A_16_s42_kscr2')]
DRAWS = 16
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


def rotate90(kspace):
    """Real-view coil k-space [B, C, N, N, 2] rotated 90 degrees about its DC sample."""
    coil = torch.view_as_complex(kspace.contiguous())
    energy = coil.abs().pow(2).sum(dim=1)[0]
    row, col = divmod(int(energy.argmax()), energy.shape[-1])
    size = energy.shape[-1]
    if row != col or energy.shape[-2] != size:
        raise SystemExit(f'rotation about DC needs a square grid with DC on the diagonal, got DC at ({row}, {col})')
    rotated = torch.roll(torch.rot90(coil, 1, dims=(-2, -1)), shifts=2 * row - (size - 1), dims=-2)
    return torch.view_as_real(rotated).contiguous()


def rotate_map(image):
    """A [N, N] k-space map rotated as `rotate90` rotates k-space (DC at the grid centre row)."""
    size = image.shape[-1]
    return np.roll(np.rot90(image, 1), shift=2 * (size // 2) - (size - 1), axis=0)


def expected_density(model, kspace, labels, index):
    """Mean final mask [N, N] over DRAWS draws of the arm's whole trajectory."""
    batch = kspace.expand(DRAWS, *kspace.shape[1:]).contiguous()
    steps = [step_thresholds(seed, [index], model.num_step, model.shape, DEVICE) for seed in range(DRAWS)]
    thresholds = [torch.cat([seed_steps[step] for seed_steps in steps]) for step in range(model.num_step)]
    finals = model.sample_trajectory(batch, labels.expand(DRAWS), thresholds)[0][-1]
    return finals.mean(dim=0)[0].float().cpu().numpy()


def correlation(a, b, region):
    return float(np.corrcoef(a[region], b[region])[0, 1])


@torch.no_grad()
def main(out):
    reference = np.load(SCORES / 'T_1_s42_kscr2.npz', allow_pickle=True)
    cfg = OmegaConf.load(DATA / 'multistep_kscr3/F_4_s42_kscr3/.hydra/config.yaml')
    cfg.datasets.num_workers = 0
    _, val_loader = build_dataloaders(cfg)
    dataset, collate = val_loader.dataset, val_loader.collate_fn
    indices = {label: int(reference['dataset_index'][row]) for label, row in ROWS.items()}
    results, epochs, images = {}, {}, {}
    for name, run in ARMS:
        model, options, epoch = load_run_copy(run)
        epochs[name] = epoch + 1
        for label, index in indices.items():
            part = ev.labelled(collate([dataset[index]]), options, DEVICE)
            centre, valid = model._initial_masks(part['kspace'])
            original = expected_density(model, part['kspace'], part['labels'], index)
            rotated = expected_density(model, rotate90(part['kspace']), part['labels'], index)
            region = (valid[0, 0] > 0).cpu().numpy() & (centre[0, 0] == 0).cpu().numpy()
            region &= rotate_map(region)
            smooth = {key: uniform_filter(value, size=SMOOTH) for key, value in
                      (('original', original), ('rotated_input', rotated), ('original_rotated', rotate_map(original)))}
            if name == ARMS[0][0]:
                images[label] = part['target'][0, 0].float().cpu().numpy()
            results[name, label] = {**smooth,
                                    'follows': correlation(smooth['rotated_input'], smooth['original_rotated'], region),
                                    'stays': correlation(smooth['rotated_input'], smooth['original'], region)}
            print(f"{name:30s} {label:22s} follows the image {results[name, label]['follows']:+.3f}   "
                  f"stays put {results[name, label]['stays']:+.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    image = images[FIGURE_SLICE]
    fig, axes = plt.subplots(len(ARMS) + 1, 2, figsize=(8.6, 4.3 * len(ARMS) + 3.6),
                             gridspec_kw={'height_ratios': [0.75] + [1] * len(ARMS)})
    for c, (picture, title) in enumerate(((image, 'Original slice'), (np.rot90(image), 'Slice rotated 90°'))):
        axes[0, c].imshow(picture, cmap='gray')
        axes[0, c].set_title(title, fontsize=13)
    axes[0, 0].set_ylabel('Input image', fontsize=12)
    for r, (name, _) in enumerate(ARMS, start=1):
        result = results[name, FIGURE_SLICE]
        for c, key in enumerate(('original', 'rotated_input')):
            axes[r, c].imshow(result[key], cmap='magma', vmin=0, vmax=0.5, interpolation='nearest')
        axes[r, 0].set_ylabel(f'{name}\nepoch {epochs[name]}', fontsize=12)
        axes[r, 1].set_xlabel(f"correlation with the original density: {result['stays']:.2f}", fontsize=11)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f'{FIGURE_SLICE} slice from the lab-meeting slide: expected sampling density\n'
                 f'(mean of {DRAWS} draws, {SMOOTH}x{SMOOTH} smoothed, colour 0 to 0.5)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=110)
    print(f'saved {out}')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
