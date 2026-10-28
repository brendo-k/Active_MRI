import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from scripts.plots.subject001_recons import AXIAL_INDEX, SCANS, axial, closest_slice, load_volume

REFERENCE = "FIRE (CG)"
COMPARED = ["Poisson", "FIRE (CG)"]
REGIONS = {
    "Frontal white matter": (45, 90, 40, 120),
    "Deep white matter / thalamus": (100, 145, 45, 115),
    "Lateral sulcus (image left)": (100, 140, 15, 60),
    "Posterior horns": (140, 185, 50, 110),
}
BOX_COLOR = "tab:red"


def crop(img, box):
    r0, r1, c0, c1 = box
    return img[r0:r1, c0:c1]


if __name__ == "__main__":
    volumes = {title: load_volume(SCANS[title]) for title in COMPARED}
    reference_slice = np.take(volumes[REFERENCE], AXIAL_INDEX, axis=2)
    images = {}
    for title, volume in volumes.items():
        index = AXIAL_INDEX if title == REFERENCE else closest_slice(volume, reference_slice, axis=2)
        images[title] = axial(volume, index)
    vmax = {title: np.percentile(img, 99.5) for title, img in images.items()}

    fig = plt.figure(figsize=(12, 3.2 * len(REGIONS)))
    grid = fig.add_gridspec(len(REGIONS), 3, width_ratios=[1.1, 1, 1])
    overview = fig.add_subplot(grid[:, 0])
    overview.imshow(images[REFERENCE], cmap="gray", vmin=0, vmax=vmax[REFERENCE])
    overview.set_xticks([])
    overview.set_yticks([])
    overview.set_title(f"{REFERENCE} axial {AXIAL_INDEX}")
    for row, (name, box) in enumerate(REGIONS.items()):
        r0, r1, c0, c1 = box
        overview.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, edgecolor=BOX_COLOR, linewidth=1.5))
        overview.text(c0, r0 - 2, str(row + 1), color=BOX_COLOR, fontsize=12, fontweight="bold")
        for col, title in enumerate(COMPARED):
            ax = fig.add_subplot(grid[row, col + 1])
            ax.imshow(crop(images[title], box), cmap="gray", vmin=0, vmax=vmax[title])
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{row + 1}. {name}")
    fig.tight_layout()
    fig.savefig("reports/figures/subject001_axial_regions.png", dpi=200)
