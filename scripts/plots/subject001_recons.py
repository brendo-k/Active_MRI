import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = "/export/scratch/bkadota/Subject_001_17.10.2026/recon_9007"
SCANS = {"GRAPPA": "MID00108", "Poisson": "MID00111", "FIRE (CG)": "MID00115"}
REFERENCE = "FIRE (CG)"
DISPLAY_LABELS = {"GRAPPA": "GRAPPA R=4", "Poisson": "Poisson R=4", "FIRE (CG)": "Adaptive w/ FIRE R=4"}
SAGITTAL_INDEX = 80
AXIAL_INDEX = 135
SLICE_AXIS = {"Sagittal": 0, "Axial": 2}
REFERENCE_INDEX = {"Sagittal": SAGITTAL_INDEX, "Axial": AXIAL_INDEX}


RERUN_DIR = os.environ.get("SUBJECT001_RERUN_DIR")
RERUN_TAG = os.environ.get("SUBJECT001_RERUN_TAG", "rerun")


def load_volume(mid):
    if RERUN_DIR:
        path = f"{RERUN_DIR}/out{mid[-3:]}_{RERUN_TAG}.h5"
    else:
        path = glob.glob(f"{ROOT}/meas_{mid}_*_recon.h5")[0]
    with h5py.File(path) as f:
        return np.asarray(f[next(iter(f))]["image_0/data"][0, 0]).astype(float)


def normalized_correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    norm = np.sqrt((a * a).sum() * (b * b).sum())
    return (a * b).sum() / norm if norm > 0 else -1.0


def closest_slice(volume, reference_slice, axis):
    scores = [normalized_correlation(reference_slice, np.take(volume, i, axis=axis)) for i in range(volume.shape[axis])]
    return int(np.argmax(scores))


def sagittal(volume, index):
    return np.flipud(np.rot90(volume[index], k=-1))


def axial(volume, index):
    return np.rot90(volume[:, ::-1, index], k=-1)


VIEWS = {"Sagittal": sagittal, "Axial": axial}
ZOOM_BOXES = {"Sagittal": (92, 148, 68, 124), "Axial": (110, 142, 30, 62)}
INSET_PAD_COLUMNS = 100
INSET_TOP = {"Sagittal": 160, "Axial": 160}
BOX_COLOR = "tab:red"
LABEL_FONTSIZE = 26
CMAP = plt.get_cmap("gray").copy()
CMAP.set_bad("white")


def pad_for_inset(img):
    return np.pad(img, ((0, 0), (0, INSET_PAD_COLUMNS)), constant_values=np.nan)


def add_zoom_inset(ax, img, box, top):
    r0, r1, c0, c1 = box
    ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, edgecolor=BOX_COLOR, linewidth=1.2))
    inset_width = INSET_PAD_COLUMNS - 8
    inset_height = inset_width * (r1 - r0) / (c1 - c0)
    inset = ax.inset_axes([img.shape[1] - INSET_PAD_COLUMNS + 4, top, inset_width, inset_height], transform=ax.transData)
    inset.imshow(img, cmap=CMAP, vmin=ax.images[0].norm.vmin, vmax=ax.images[0].norm.vmax, interpolation="nearest")
    inset.set_xlim(c0 - 0.5, c1 - 0.5)
    inset.set_ylim(r1 - 0.5, r0 - 0.5)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor(BOX_COLOR)
        spine.set_linewidth(1.5)

if __name__ == "__main__":
    volumes = {title: load_volume(mid) for title, mid in SCANS.items()}
    padded_width = {name: 256 + INSET_PAD_COLUMNS if name == "Sagittal" else 160 + INSET_PAD_COLUMNS for name in VIEWS}
    height_ratios = [256 / padded_width[name] for name in VIEWS]
    fig = plt.figure(figsize=(18, 5.4 * sum(height_ratios) * 1.1))
    grid = fig.add_gridspec(len(VIEWS), len(SCANS), height_ratios=height_ratios)
    for r, (view_name, view) in enumerate(VIEWS.items()):
        axis = SLICE_AXIS[view_name]
        reference_slice = np.take(volumes[REFERENCE], REFERENCE_INDEX[view_name], axis=axis)
        for c, (title, volume) in enumerate(volumes.items()):
            index = closest_slice(volume, reference_slice, axis) if title != REFERENCE else REFERENCE_INDEX[view_name]
            print(view_name, title, "slice", index)
            img = view(volume, index)
            ax = fig.add_subplot(grid[r, c])
            ax.imshow(pad_for_inset(img), cmap=CMAP, vmin=0, vmax=np.percentile(img, 99.5))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            add_zoom_inset(ax, pad_for_inset(img), ZOOM_BOXES[view_name], INSET_TOP[view_name])
            if r == 0:
                ax.set_title(DISPLAY_LABELS[title], fontsize=LABEL_FONTSIZE, pad=12)
            if c == 0:
                ax.set_ylabel(view_name, fontsize=LABEL_FONTSIZE, labelpad=12)
    fig.tight_layout()
    stem = f"reports/figures/subject001_brain{'_' + RERUN_TAG if RERUN_DIR else ''}"
    fig.savefig(f"{stem}.png", dpi=400)
    fig.savefig(f"{stem}.pdf")
