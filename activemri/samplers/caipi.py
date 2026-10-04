"""Fixed 2D-CAIPI undersampling lattice."""
import torch


def caipi_mask(shape, accel_y=2, accel_z=2, shift=1, keep_fraction=1.0):
    """Binary `[H, W]` 2D-CAIPI lattice in DC-centred (fft-shifted) layout.

    Every `accel_y`-th row is sampled, and within sampled row `r` the columns
    are every `accel_z`-th one, offset by `shift * r` columns so successive
    sampled rows are staggered. Net acceleration is `accel_y * accel_z`. DC
    (`[H // 2, W // 2]`) is always a lattice point.

    `keep_fraction < 1` is partial Fourier along the columns: lattice points in the
    rightmost `1 - keep_fraction` of columns are dropped (6/8 -> 0.75).
    """
    rows = torch.arange(shape[0]) - shape[0] // 2
    cols = torch.arange(shape[1]) - shape[1] // 2
    sampled_row = rows % accel_y == 0
    row_index = torch.div(rows, accel_y, rounding_mode='floor')
    sampled_col = (cols[None, :] - shift * row_index[:, None]) % accel_z == 0
    kept_col = torch.arange(shape[1]) < round(keep_fraction * shape[1])
    return (sampled_row[:, None] & sampled_col & kept_col[None, :]).float()
