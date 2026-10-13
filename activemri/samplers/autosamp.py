"""AutoSamp sampler: learned k-space coordinates, snapped to the Cartesian grid.

Adapted from https://arxiv.org/abs/2306.02888 (https://github.com/alkanc/AutoSamp).
AutoSamp optimizes a fixed number of continuous (ky, kx) sample positions through
a non-uniform Fourier operator. Here the acquisition must be Cartesian, so each
position is snapped to the nearest grid cell by L2 distance instead.

Positions live in cell units on the fft-shifted array (cell centres at integers,
DC at [H // 2, W // 2]) and wrap around the k-space extent, as AutoSamp wraps its
coordinates modulo the field of view.

Snapping is not differentiable, so the forward pass emits the hard mask and the
backward pass uses a bilinear splat of each continuous position onto its four
neighbouring cells: a straight-through estimator whose gradient moves a point
towards the neighbours where adding a sample would lower the loss.

That estimator traps points: it always favours the empty neighbour over the
sampled cell, so once a point crosses the x.5 snap boundary the gradient flips and
it oscillates there (reports/AUTOSAMP_REINFORCE_STALL.md). Setting `kernel_sigma`
> 0 in training swaps the bilinear surrogate for a Gaussian splat of width
`kernel_sigma`, whose gradient weighs every cell within a few widths of the point
instead of only the two it straddles; `annealed_kernel_sigma` narrows it over
training. The forward mask stays hard either way: k-space here is noiseless, so a
fractional mask weight w on a cell would reveal that cell's full value (w * y scales
straight back out) and leak samples beyond the budget.

Two points can round to the same cell, and a point can land on a cell that is
already acquired. Both would silently spend less than the budget, so each is
reassigned to the nearest free cell and, in training, the parameter is moved
there too (the analogue of AutoSamp's `ModuloConstraint`). The number of new
samples therefore equals `num_points` exactly, except where k-space is zero-padded:
a point that lands in a slice's padding is discarded by `valid_mask` like every other
sampler's, and gets no gradient there, so for padded anatomies the effective
acceleration is higher than requested.
"""
import math

import torch
from torch import nn


INIT_MODES = ("uniform", "normal")
NORMAL_INIT_STD_FRACTION = 0.25  # AutoSamp's truncated normal: std = a quarter of the extent
NORMAL_INIT_TRUNCATION = 2.0  # in standard deviations
INITIAL_SEARCH_RADIUS = 2  # cells; the search window around a point doubles while it is full
KERNEL_RADIUS_IN_SIGMAS = 3.0  # the Gaussian surrogate is truncated at this many widths


def annealed_kernel_sigma(sigma_start, sigma_end, anneal_fraction, progress):
    """Kernel width decaying geometrically from `sigma_start` to `sigma_end` over the
    first `anneal_fraction` of training (`progress` in [0, 1]); the trainer freezes the
    positions on the hard mask after that."""
    t = min(progress / anneal_fraction, 1.0) if anneal_fraction > 0 else 1.0
    return sigma_start * (sigma_end / sigma_start) ** t


class AutoSampCoordinates(nn.Module):
    """`num_points` learned sample positions over an `[H, W]` k-space grid.

    Args:
        shape: Shape of the reconstructed image.
        num_points: Number of new locations to acquire.
        init: `uniform` draws distinct cells uniformly; `normal` draws them with a
            truncated Gaussian density centred on DC (AutoSamp's variable-density
            initializer).
    """

    def __init__(self, shape=(320, 320), num_points=1, init="uniform"):
        super().__init__()
        if init not in INIT_MODES:
            raise ValueError(f"Unknown autosamp init {init!r}; expected one of {INIT_MODES}")
        if not 0 < num_points <= shape[0] * shape[1]:
            raise ValueError(f"num_points={num_points} must be in (0, {shape[0] * shape[1]}]")

        self.shape = tuple(shape)
        self.num_points = num_points
        self.positions = nn.Parameter(self._initial_positions(init))
        self.register_buffer("previous_cells", torch.full((num_points,), -1, dtype=torch.long), persistent=False)
        self.cell_change_fraction = 0.0
        self.kernel_sigma = 0.0  # cells; > 0 takes the training gradient from a Gaussian surrogate

    def forward(self, old_mask, valid_mask=None):
        """Args:
            old_mask: [B, 1, H, W] locations already acquired.
            valid_mask: Optional [B, 1, H, W] in-bounds k-space; samples that land
                outside it are discarded, not re-spent, as for every other sampler.

        Returns:
            new_mask: [1, 1, H, W] (or [B, ...] if `valid_mask` is batched) of the
                newly acquired locations: the hard mask, with a straight-through
                gradient to `positions` from the bilinear surrogate, or in training with
                `kernel_sigma` > 0 the Gaussian one.
            soft_mask: [H, W] the splat that carries that gradient, for display.
        """
        height, width = self.shape
        taken = old_mask.amax(dim=(0, 1)) > 0
        free = (~taken).to(self.positions.dtype)

        cells, moved_positions = self._snap(self.positions.detach(), taken)
        if self.training:
            with torch.no_grad():
                self.positions.copy_(self._wrap(moved_positions))
        self._record_cell_changes(cells)

        hard = torch.zeros(height * width, device=cells.device, dtype=self.positions.dtype)
        hard[cells] = 1.0
        if self.training and self.kernel_sigma > 0:
            soft = self._gaussian_splat(self._wrap(self.positions), self.kernel_sigma) * free
        else:
            soft = self._splat(self._wrap(self.positions)) * free
        new_mask = (hard.view(height, width) + (soft - soft.detach()))[None, None]

        if valid_mask is not None:
            new_mask = new_mask * valid_mask.to(new_mask.dtype)
        return new_mask, soft.detach()

    def _initial_positions(self, init):
        height, width = self.shape
        rows = (torch.arange(height) - height // 2).float()[:, None].expand(height, width)
        cols = (torch.arange(width) - width // 2).float()[None, :].expand(height, width)
        if init == "uniform":
            logits = torch.zeros(height, width)
        else:
            z_row = rows / (NORMAL_INIT_STD_FRACTION * height)
            z_col = cols / (NORMAL_INIT_STD_FRACTION * width)
            logits = -0.5 * (z_row ** 2 + z_col ** 2)
            logits = logits.masked_fill((z_row.abs() > NORMAL_INIT_TRUNCATION) | (z_col.abs() > NORMAL_INIT_TRUNCATION), float("-inf"))
        # Gumbel top-k draws num_points distinct cells with probability following `logits`.
        scores = logits.flatten() + torch.distributions.Gumbel(0.0, 1.0).sample((height * width,))
        chosen = scores.topk(self.num_points).indices
        return torch.stack([chosen // width, chosen % width], dim=1).float()

    def _wrap(self, positions):
        """Fold positions into [-0.5, size - 0.5), the span of cells that round into the grid."""
        size = positions.new_tensor(self.shape)
        return torch.remainder(positions + 0.5, size) - 0.5

    @torch.no_grad()
    def _snap(self, positions, taken):
        """Nearest-free-cell assignment, one distinct cell per point.

        Each point claims the free cell nearest to it in L2. When several points
        claim the same cell the nearest one keeps it (lowest index on a tie) and the
        others search again with that cell removed, widening the window whenever a
        point finds none free.

        Returns:
            cells: [K] flat cell index of every point.
            moved_positions: [K, 2] the positions, with every point that had to be
                reassigned moved onto the centre of its new cell and the rest untouched.
        """
        height, width = self.shape
        size = positions.new_tensor(self.shape)
        free = ~taken.flatten()
        if int(free.sum()) < self.num_points:
            raise ValueError(f"{self.num_points} points requested but only {int(free.sum())} cells are free")

        positions = self._wrap(positions)
        centre = positions.round()
        own_cell = torch.remainder(centre[:, 0], height).long() * width + torch.remainder(centre[:, 1], width).long()
        cells = torch.full((self.num_points,), -1, dtype=torch.long, device=positions.device)
        moved_positions = positions.clone()
        pending = torch.arange(self.num_points, device=positions.device)
        radius = INITIAL_SEARCH_RADIUS

        while pending.numel():
            span = torch.arange(-radius, radius + 1, device=positions.device)
            offsets = torch.stack(torch.meshgrid(span, span, indexing="ij"), dim=-1).view(-1, 2).to(positions.dtype)
            candidates = centre[pending, None, :] + offsets[None]  # [P, M, 2], unwrapped
            distance = (candidates - positions[pending, None, :]).pow(2).sum(dim=-1)
            wrapped = torch.remainder(candidates, size).long()
            flat = wrapped[..., 0] * width + wrapped[..., 1]
            distance = distance.masked_fill(~free[flat], float("inf"))

            best_distance, best = distance.min(dim=1)
            claimed = flat.gather(1, best[:, None]).squeeze(1)
            reachable = torch.isfinite(best_distance)
            if not reachable.any():
                radius *= 2
                continue

            # The nearest claimant keeps a contested cell; ties go to the lowest point index.
            nearest = torch.full((height * width,), float("inf"), device=positions.device)
            nearest.scatter_reduce_(0, claimed[reachable], best_distance[reachable], "amin")
            closest = reachable & (best_distance == nearest[claimed])
            lowest = torch.full((height * width,), self.num_points, device=positions.device)
            lowest.scatter_reduce_(0, claimed[closest], pending[closest], "amin")
            won = closest & (pending == lowest[claimed])

            winners = pending[won]
            cells[winners] = claimed[won]
            reassigned = claimed[won] != own_cell[winners]
            moved_positions[winners[reassigned]] = candidates[won, best[won]][reassigned]
            free[claimed[won]] = False
            pending = pending[~won]
            if not reachable.all():
                radius *= 2

        return cells, moved_positions

    def _splat(self, positions):
        """Bilinear weight of each point on its four neighbouring cells, summed into an [H, W] map."""
        height, width = self.shape
        floor = positions.detach().floor()
        frac = positions - floor
        floor = floor.long()
        weights, flat = [], []
        for dy in (0, 1):
            for dx in (0, 1):
                wy = frac[:, 0] if dy else 1 - frac[:, 0]
                wx = frac[:, 1] if dx else 1 - frac[:, 1]
                row = torch.remainder(floor[:, 0] + dy, height)
                col = torch.remainder(floor[:, 1] + dx, width)
                weights.append(wy * wx)
                flat.append(row * width + col)
        splat = torch.zeros(height * width, device=positions.device, dtype=positions.dtype)
        splat.index_add_(0, torch.cat(flat), torch.cat(weights))
        return splat.view(height, width)

    def _gaussian_splat(self, positions, sigma):
        """Each point's unit mass spread over the cells within `KERNEL_RADIUS_IN_SIGMAS`
        widths of it, weighted by a Gaussian of width `sigma` (cells), summed into an [H, W] map."""
        height, width = self.shape
        radius = min(max(1, math.ceil(KERNEL_RADIUS_IN_SIGMAS * sigma)), (min(height, width) - 1) // 2)
        span = torch.arange(-radius, radius + 1, device=positions.device)
        offsets = torch.stack(torch.meshgrid(span, span, indexing="ij"), dim=-1).view(-1, 2).to(positions.dtype)
        candidates = positions.detach().round()[:, None, :] + offsets[None]  # [K, M, 2], unwrapped
        distance = (candidates - positions[:, None, :]).pow(2).sum(dim=-1)
        weights = torch.softmax(-distance / (2 * sigma ** 2), dim=1)

        wrapped = torch.remainder(candidates, positions.new_tensor(self.shape)).long()
        flat = wrapped[..., 0] * width + wrapped[..., 1]
        splat = torch.zeros(height * width, device=positions.device, dtype=positions.dtype)
        splat.index_add_(0, flat.flatten(), weights.flatten())
        return splat.view(height, width)

    def _record_cell_changes(self, cells):
        """Fraction of points on a different cell than at the previous forward pass."""
        seen_before = self.previous_cells >= 0
        changed = (self.previous_cells != cells) & seen_before
        self.cell_change_fraction = changed.sum().item() / max(int(seen_before.sum()), 1)
        self.previous_cells.copy_(cells)
