"""Oracle sampling-pattern search using CG-SENSE as the reconstructor.

The reconstructor is a *fixed linear* operator: CG-SENSE solves

    (A^H A + lam I) x = A^H y,     A = M F S

so for a given mask the reconstruction is completely determined -- no network,
no training. That makes it usable as an oracle: score a candidate mask by the
error of the image it yields, and search directly over masks.

Two regimes, with very different costs:

1D (whole readout lines).  A line mask varies only along the phase-encode axis,
so ``F^H M F`` is a 1D convolution along that axis and commutes with the
readout transform.  ``A^H A`` is then exactly block diagonal in the readout
index, with block ``a``

    G_a = Cop * R_a,        R_a[i, j] = sum_c conj(S[c, a, i]) S[c, a, j]

a Hadamard product of a mask-only factor ``Cop`` (Toeplitz, rank-1 update per
added line) with a mask-independent Gram factor ``R_a``.  Every block is W x W,
so the "iterative" solve becomes a batch of small exact Cholesky solves and the
greedy search is tractable in closed form.

2D (arbitrary k-space points).  The separability is gone: ``A^H A`` is a dense
HW x HW operator that cannot be formed.  Greedy still works, but the candidate
score has to be evaluated matrix-free -- see :func:`greedy_2d`.
"""
from typing import Dict, List, Optional, Sequence

import torch

from activemri.utils.image_processing import fft_2d_img, ifft_2d_img

PE_DIM = -1  # phase encode: the axis a 1D line mask varies along (matches LineConstrainedProbMask)


# --------------------------------------------------------------------------
# 1D: exact block-diagonal machinery
# --------------------------------------------------------------------------

def _fft_pe(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    x = torch.fft.ifftshift(x, dim=PE_DIM)
    x = (torch.fft.ifft if inverse else torch.fft.fft)(x, dim=PE_DIM, norm='ortho')
    return torch.fft.fftshift(x, dim=PE_DIM)


def line_conv_operator(line_mask: torch.Tensor) -> torch.Tensor:
    """The W x W operator ``F_pe^H diag(line_mask) F_pe``.

    Args:
        line_mask: [W] real, 1 for a sampled phase-encode line.

    Returns:
        [W, W] complex Toeplitz operator.
    """
    W = line_mask.shape[-1]
    eye = torch.eye(W, dtype=torch.complex64, device=line_mask.device)
    return _fft_pe(line_mask.to(eye.dtype) * _fft_pe(eye), inverse=True).transpose(-1, -2)


def line_conv_basis(W: int, device) -> torch.Tensor:
    """Per-line contributions to ``Cop``, so a mask's operator is a plain sum.

    ``line_conv_operator(m) == (m[:, None, None] * basis).sum(0)``, which makes
    adding a line a rank-1 update rather than a fresh transform.

    Returns:
        [W, W, W] complex; entry k is the operator contributed by line k alone.
    """
    return torch.stack([
        line_conv_operator(torch.nn.functional.one_hot(torch.tensor(k), W).float().to(device))
        for k in range(W)
    ])


def estimate_sense_maps(kspace: torch.Tensor, acs: int = 24, threshold: float = 0.05):
    """Estimate coil sensitivities from the centre of the *preprocessed* k-space.

    The stored ESPIRiT maps are computed on the raw grid, so they no longer line
    up once the loader decimates and crops k-space. Re-estimating from the same
    array the oracle reconstructs from keeps the forward model self-consistent,
    which is what matters when the mask is the thing under study.

    Low-resolution coil images normalised by their root-sum-of-squares, i.e. the
    standard ACS estimate, with air zeroed so it cannot dominate a noise
    criterion.

    Args:
        kspace: [C, H, W] complex fully sampled k-space.
        acs: Side length of the centre block used for the estimate.
        threshold: Fraction of the peak RSS below which a pixel counts as air.

    Returns:
        ``maps`` [C, H, W] complex with ``sum_c |S_c|^2`` equal to 1 on the
        object and 0 outside, and the ``support`` mask [H, W].
    """
    C, H, W = kspace.shape
    centre = torch.zeros(H, W, device=kspace.device)
    centre[H // 2 - acs // 2: H // 2 + acs // 2, W // 2 - acs // 2: W // 2 + acs // 2] = 1.0

    low_res = ifft_2d_img(kspace * centre)
    rss = low_res.abs().pow(2).sum(dim=0).sqrt()
    support = (rss > threshold * rss.max()).to(rss.dtype)
    maps = low_res / rss.clamp(min=1e-12) * support
    return maps, support


def sense_gram_blocks(sense_maps: torch.Tensor) -> torch.Tensor:
    """Mask-independent Gram factor ``R_a`` of every readout block.

    Args:
        sense_maps: [C, H, W] complex coil sensitivities.

    Returns:
        [H, W, W] complex, Hermitian per block.
    """
    return torch.einsum('cai,caj->aij', sense_maps.conj(), sense_maps)


class LineOracle:
    """Exact CG-SENSE oracle over 1D (line) sampling masks for one slice.

    Every quantity that does not depend on the mask is precomputed once, so
    scoring a candidate mask costs one batched Cholesky solve over the H
    readout blocks.

    Args:
        kspace: [C, H, W] complex fully sampled k-space.
        sense_maps: [C, H, W] complex coil sensitivities.
        lam: Tikhonov weight, relative to the mean diagonal of the fully
            sampled normal operator.
        noise_level: Noise strength, calibrated so that this is exactly the
            NMSE the *fully sampled* reconstruction suffers from noise alone.
            That puts it on the same scale as the aliasing error, so 0.02 means
            "noise costs as much as a 2% NMSE at R=1". Only ``'mse_noise'``
            uses it, and it is what stops the search from collapsing onto a
            pure low-pass block.
        support: [H, W] weight restricting every criterion to the object.
            Without it the air outside the patient contributes a large,
            mask-independent noise term that swamps the signal.
    """

    def __init__(self, kspace: torch.Tensor, sense_maps: torch.Tensor, lam: float = 1e-3,
                 noise_level: float = 0.0, support: Optional[torch.Tensor] = None):
        self.device = kspace.device
        self.C, self.H, self.W = kspace.shape

        self.sense_maps = sense_maps
        self.coil_imgs = ifft_2d_img(kspace)                       # [C, H, W]
        self.R = sense_gram_blocks(sense_maps)                     # [H, W, W]
        self.basis = line_conv_basis(self.W, self.device)          # [W, W, W]

        if support is None:
            support = (torch.diagonal(self.R, dim1=-2, dim2=-1).real > 1e-6).float()
        self.support = support                                     # [H, W]

        diag = torch.diagonal(self.R, dim1=-2, dim2=-1).real       # [H, W] = sum_c |S|^2
        self.lam = lam * diag.mean().clamp(min=1e-12)
        self.eye = torch.eye(self.W, dtype=torch.complex64, device=self.device)

        # Fully sampled reference: Cop = I, so G is diagonal and the solve is exact.
        b_full = torch.einsum('cai,cai->ai', sense_maps.conj(), self.coil_imgs)
        self.x_ref = b_full / (diag + self.lam).to(b_full.dtype) * support   # [H, W]
        self.ref_energy = self.x_ref.abs().pow(2).sum().clamp(min=1e-12)

        # Calibrate the noise so `noise_level` reads directly as the NMSE that
        # noise alone costs at full sampling.
        ones = torch.ones(1, self.W, device=self.device)
        full_var = self._variance(self._systems(ones)[0]).clamp(min=1e-12)
        self.noise_var = noise_level * self.ref_energy / full_var

    def _systems(self, masks: torch.Tensor):
        """Build ``(G, b)`` for a batch of line masks.

        Args:
            masks: [N, W] real line masks.

        Returns:
            G [N, H, W, W] and b [N, H, W].
        """
        cop = torch.einsum('nk,kij->nij', masks.to(self.basis.dtype), self.basis)   # [N, W, W]
        G = cop.unsqueeze(1) * self.R.unsqueeze(0)                                  # [N, H, W, W]
        G = G + self.lam * self.eye
        masked_coil_imgs = torch.einsum('nij,caj->ncai', cop, self.coil_imgs)       # [N, C, H, W]
        b = torch.einsum('cai,ncai->nai', self.sense_maps.conj(), masked_coil_imgs)
        return G, b

    def reconstruct(self, masks: torch.Tensor) -> torch.Tensor:
        """Exact regularised SENSE reconstruction for each line mask.

        Args:
            masks: [N, W] real line masks.

        Returns:
            [N, H, W] complex images.
        """
        G, b = self._systems(masks)
        return torch.linalg.solve(G, b.unsqueeze(-1)).squeeze(-1)

    def score(self, masks: torch.Tensor, objective: str = 'mse') -> torch.Tensor:
        """Score line masks; lower is better.

        Args:
            masks: [N, W] real line masks.
            objective: ``'mse'`` is the image-dependent oracle, the normalised
                squared error against the fully sampled reconstruction.
                ``'trace'`` (A-optimality) and ``'logdet'`` (D-optimality) are
                image-independent noise-amplification criteria that need only
                the sensitivity maps.

        Returns:
            [N] real scores.
        """
        if objective == 'mse':
            return self._bias(masks) / self.ref_energy

        if objective == 'mse_noise':
            G, b = self._systems(masks)
            x = torch.linalg.solve(G, b.unsqueeze(-1)).squeeze(-1)
            bias = ((x - self.x_ref).abs().pow(2) * self.support).sum(dim=(-2, -1))
            return (bias + self.noise_var * self._variance(G)) / self.ref_energy

        G, _ = self._systems(masks)
        if objective == 'trace':
            return self._variance(G)
        if objective == 'logdet':
            chol = torch.linalg.cholesky(G.to(torch.complex128))
            return -2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1).real
                                    .clamp(min=1e-30)).sum(dim=(-2, -1)).to(torch.float32)
        raise ValueError(f"Unknown objective '{objective}'")

    def _bias(self, masks: torch.Tensor) -> torch.Tensor:
        x = self.reconstruct(masks)
        return ((x - self.x_ref).abs().pow(2) * self.support).sum(dim=(-2, -1))

    def _variance(self, G: torch.Tensor) -> torch.Tensor:
        """Noise power of the estimator, ``tr(G^-1 A^H A G^-1)``, per mask.

        Uses ``A^H A = G - lam I`` so the whole thing stays inside the
        block-diagonal algebra, and inverts through a Cholesky factor in float64
        rather than forming ``G^-1`` directly -- the blocks are near singular
        wherever the coil sensitivities vanish. Only pixels inside ``support``
        are counted.

        Args:
            G: [N, H, W, W] regularised normal operator blocks.

        Returns:
            [N] real noise power, summed over all blocks and pixels.
        """
        chol = torch.linalg.cholesky(G.to(torch.complex128))
        eye = torch.eye(G.shape[-1], dtype=chol.dtype, device=G.device).expand_as(chol)
        inv_chol = torch.linalg.solve_triangular(chol, eye, upper=False)
        G_inv = inv_chol.conj().transpose(-1, -2) @ inv_chol

        var = torch.diagonal(G_inv, dim1=-2, dim2=-1).real  # [N, H, W]
        var = var - self.lam * G_inv.abs().pow(2).sum(dim=-2)  # diag(G^-2), G^-1 Hermitian
        return (var * self.support).sum(dim=(-2, -1)).to(torch.float32)

    def score_chunked(self, masks: torch.Tensor, objective: str, chunk: int) -> torch.Tensor:
        return torch.cat([
            self.score(masks[i:i + chunk], objective) for i in range(0, masks.shape[0], chunk)
        ])


class MeanLineOracle:
    """Averages :class:`LineOracle` scores over several slices.

    A single-slice oracle finds the mask that is best for that one image, which
    is exactly the quantity that does not generalise. Averaging the score over a
    set of slices before the greedy pick turns it into a population oracle.

    Args:
        oracles: Per-slice oracles, all sharing a phase-encode width.
    """

    def __init__(self, oracles: Sequence[LineOracle]):
        if len({o.W for o in oracles} | {o.W for o in oracles}) != 1:
            raise ValueError('All oracles must share the same phase-encode width')
        self.oracles = list(oracles)
        self.W = self.oracles[0].W
        self.device = self.oracles[0].device

    def score(self, masks: torch.Tensor, objective: str = 'mse') -> torch.Tensor:
        return torch.stack([o.score(masks, objective) for o in self.oracles]).mean(dim=0)

    def score_chunked(self, masks: torch.Tensor, objective: str, chunk: int) -> torch.Tensor:
        return torch.cat([
            self.score(masks[i:i + chunk], objective) for i in range(0, masks.shape[0], chunk)
        ])


def center_lines(W: int, num: int, device) -> torch.Tensor:
    """A DC-centred ACS block of ``num`` lines, matching the data's fftshift convention."""
    mask = torch.zeros(W, device=device)
    if num > 0:
        mask[W // 2 - num // 2: W // 2 - num // 2 + num] = 1.0
    return mask


def greedy_lines(
    oracle: LineOracle,
    budget: int,
    objective: str = 'mse',
    preselect: int = 0,
    chunk: int = 32,
    verbose: bool = True,
) -> Dict:
    """Greedily add the phase-encode line that most improves the oracle score.

    Args:
        oracle: A :class:`LineOracle` for the slice being fitted.
        budget: Total number of sampled lines, ACS included.
        objective: Passed to :meth:`LineOracle.score`.
        preselect: Number of centre lines fixed before the search starts.
        chunk: Candidate masks scored per batched solve; trades memory for speed.
        verbose: Print progress.

    Returns:
        ``mask`` [W], ``order`` of chosen lines, and the score ``history``.
    """
    W = oracle.W
    mask = center_lines(W, preselect, oracle.device)
    order = mask.nonzero(as_tuple=False).flatten().tolist()
    history = [oracle.score(mask.unsqueeze(0), objective).item()]

    while int(mask.sum().item()) < budget:
        candidates = (mask == 0).nonzero(as_tuple=False).flatten()
        trial = mask.unsqueeze(0).repeat(candidates.shape[0], 1)
        trial[torch.arange(candidates.shape[0]), candidates] = 1.0

        scores = oracle.score_chunked(trial, objective, chunk)
        best = int(torch.argmin(scores).item())

        mask[candidates[best]] = 1.0
        order.append(int(candidates[best].item()))
        history.append(float(scores[best].item()))
        if verbose:
            print(f'  lines={int(mask.sum().item()):3d}  picked={order[-1]:3d}  '
                  f'{objective}={history[-1]:.6g}', flush=True)

    return {'mask': mask, 'order': order, 'history': history}


def plackett_luce_sample_and_logprob(logits: torch.Tensor, k: int):
    """Gumbel-top-k sampling from a Plackett-Luce policy, with its log-probability.

    Sampling ``k`` items without replacement in proportion to ``softmax(logits)``,
    sequentially, is exactly Gumbel-top-k: add i.i.d. Gumbel noise to every logit
    and take the top ``k`` (Kool et al., 2019). That gives the sample in one
    vectorised step instead of a Python loop; only the log-probability -- which
    a policy-gradient update needs -- still requires walking the chosen order
    once, since it is a product of ``k`` categorical draws over shrinking
    candidate sets: ``sum_t [ logit_{i_t} - logsumexp(logits over items not yet
    taken) ]``.

    Args:
        logits: [B, N] unnormalised scores over candidates.
        k: Number of items to select per row, without replacement.

    Returns:
        ``order`` [B, k] selected indices (highest-probability first) and
        ``logprob`` [B] the Plackett-Luce log-probability of that order.
    """
    B, N = logits.shape
    # clamp_min binds to torch.log(u), which is already <=0 for u in [0,1) -- clamping
    # a non-positive number up to 1e-12 saturates it to exactly 1e-12 every time, so the
    # outer log(-1e-12) was NaN on every single call. Clamping u itself away from both
    # 0 and 1 keeps both logs finite without that degenerate saturation.
    u = torch.rand_like(logits).clamp(1e-12, 1 - 1e-12)
    gumbel = -torch.log(-torch.log(u))
    order = torch.topk(logits + gumbel, k, dim=1).indices  # [B, k]

    selected_logits = torch.gather(logits, 1, order)  # [B, k], in selection order
    # logsumexp over "not yet taken" at step t: mask out items taken at steps < t.
    # Since order lists items in the order they were taken, an out-of-place OR
    # with each step's one-hot reproduces exactly that shrinking candidate set --
    # in-place accumulation would mutate the mask a later backward pass needs to
    # replay masked_fill's forward at each earlier step.
    taken = torch.zeros(B, N, dtype=torch.bool, device=logits.device)
    logp = torch.zeros(B, device=logits.device)
    for t in range(k):
        masked = logits.masked_fill(taken, float('-inf'))
        logp = logp + selected_logits[:, t] - torch.logsumexp(masked, dim=1)
        taken = taken | torch.nn.functional.one_hot(order[:, t], N).bool()
    return order, logp


def reinforce_lines(
    oracle,
    budget: int,
    objective: str = 'mse_noise',
    preselect: int = 0,
    num_episodes: int = 300,
    batch_size: int = 32,
    lr: float = 0.1,
    baseline_momentum: float = 0.9,
    chunk: int = 32,
    verbose: bool = True,
) -> Dict:
    """Learns a sequential line-selection policy by REINFORCE against the oracle.

    The natural RL framing of this problem (as in DDQN-style active-acquisition
    baselines) is an MDP that picks one line at a time, rewarded by how much the
    reconstruction improves. Here the same sequential-without-replacement action
    space is used, but the policy is a single shared logit per line -- a
    Plackett-Luce distribution over *orderings* -- rather than a per-step network
    conditioned on the running reconstruction; that keeps training a matter of
    scoring whole sampled masks with :class:`LineOracle` (cheap, closed form)
    instead of a step-by-step CG solve per action. It is the policy-gradient
    analogue of :func:`loupe_style_design`: both learn one distribution over
    lines from oracle feedback, but this one keeps the hard combinatorial
    constraint exactly (every episode samples a real ``budget``-line mask) rather
    than relaxing to a continuous ``p`` and rounding at the end.

    Args:
        oracle: A :class:`LineOracle` or :class:`MeanLineOracle`.
        budget: Total number of sampled lines, ACS included.
        objective: Passed to :meth:`LineOracle.score`; reward is ``-score``.
        preselect: Number of centre lines fixed before the search starts.
        num_episodes: Policy-gradient update steps.
        batch_size: Episodes (sampled masks) per update, for a lower-variance
            gradient estimate.
        lr: Adam learning rate on the logits.
        baseline_momentum: EMA decay for the REINFORCE baseline, which is
            plain mean-reward subtraction -- simple and sufficient here since
            all episodes in a batch share one policy.
        chunk: Candidate masks scored per batched solve.
        verbose: Print progress.

    Returns:
        ``mask`` [W] (argmax rollout under the learned logits), ``logits`` [W],
        and the per-episode mean-reward ``history``.
    """
    W = oracle.W
    device = oracle.device
    center = center_lines(W, preselect, device)
    free_idx = center.eq(0).nonzero(as_tuple=False).flatten()
    k_free = budget - preselect
    if k_free < 0:
        raise ValueError('budget must be >= preselect')

    logits = torch.zeros(free_idx.numel(), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)
    baseline = None
    history = []

    for ep in range(num_episodes):
        order, logp = plackett_luce_sample_and_logprob(logits.unsqueeze(0).expand(batch_size, -1), k_free)
        masks = center.unsqueeze(0).repeat(batch_size, 1)
        rows = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, k_free)
        masks[rows, free_idx[order]] = 1.0

        with torch.no_grad():
            scores = oracle.score_chunked(masks, objective, chunk)  # [B], lower is better
        reward = -scores
        baseline = reward.mean() if baseline is None else baseline_momentum * baseline + (1 - baseline_momentum) * reward.mean()
        advantage = reward - baseline

        loss = -(advantage.detach() * logp).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(scores.mean().item())
        if verbose and (ep % 25 == 0 or ep == num_episodes - 1):
            print(f'  episode={ep:4d}  mean_{objective}={history[-1]:.6g}  best_in_batch={scores.min().item():.6g}',
                  flush=True)

    with torch.no_grad():
        final_idx = torch.topk(logits, k_free).indices
        mask = center.clone()
        mask[free_idx[final_idx]] = 1.0
    return {'mask': mask, 'logits': logits.detach(), 'history': history}


def project_capped_simplex(p: torch.Tensor, budget: float, iters: int = 60) -> torch.Tensor:
    """Euclidean projection onto ``{p : 0 <= p_i <= 1, sum(p) = budget}``.

    Standard bisection on the shift ``tau`` in ``clamp(p - tau, 0, 1)``: the sum
    of the clamped vector is monotone decreasing in ``tau``, so its root (where
    the sum hits ``budget``) is found by bisection rather than a sort.

    Args:
        p: [W] real, any range.
        budget: Target sum, ``0 <= budget <= W``.
        iters: Bisection steps; 60 gives machine precision for float32.

    Returns:
        [W] real in ``[0, 1]`` summing to ``budget``.
    """
    lo, hi = (p.min() - 1).item(), (p.max()).item()
    for _ in range(iters):
        tau = (lo + hi) / 2
        if (p - tau).clamp(0, 1).sum().item() > budget:
            lo = tau
        else:
            hi = tau
    return (p - tau).clamp(0, 1)


def _projected_gradient_core(
    value_at,
    N: int,
    budget: int,
    num_steps: int,
    lr: float,
    eps: float,
    init: Optional[torch.Tensor],
    verbose: bool,
    device,
    log_prefix: str = '',
    label: str = 'score',
):
    """Backtracking projected gradient descent on ``p in {0<=p<=1, sum p = budget}``.

    Shared machinery for :func:`relaxed_design` / :func:`loupe_style_design`
    (1D, through the closed-form :class:`LineOracle`) and their 2D analogues
    (through a differentiable CG-SENSE solve). Nothing here is specific to
    how ``value_at`` is computed or whether it is convex -- convexity only
    changes what the returned local optimum is *worth*, not how it is found
    -- so both regimes share one implementation and one set of numerical
    fixes (see the tolerance/step-cap comments below).

    Args:
        value_at: Callable ``p -> scalar tensor``, already clamped to
            ``(eps, 1-eps)`` by the caller's closure; differentiable in ``p``.
        N: Dimensionality of ``p``.
        device: Device ``p`` lives on.
        label: Name used in the progress line (e.g. the objective name).
    """
    if init is None:
        # Uniform p is an exact stationary point for 'trace': every DFT column has
        # constant magnitude 1/sqrt(W), so at p=uniform every block collapses to a
        # diagonal matrix and the (projected) gradient is identical at every k --
        # i.e. exactly zero after removing the mean component for the sum
        # constraint. A tiny random perturbation breaks that saddle.
        init = torch.full((N,), budget / N) + 1e-3 * torch.randn(N)
    else:
        # A hard 0/1 warm start (e.g. a greedy mask) sits exactly on the box
        # boundary, where `clamp` in value_at has zero gradient by construction
        # -- the search would see grad=0 and immediately declare convergence
        # without ever looking around. Nudging into the interior restores a
        # real gradient while leaving sum(p) essentially unchanged (the
        # projection below fixes it exactly).
        init = init * (1 - 2 * eps) + eps
    p = project_capped_simplex(init, budget).to(device).requires_grad_(True)
    step_size = lr
    max_step_size = 4 * lr
    loss = value_at(p)
    # float32 score magnitudes here are often ~1e-2 to 1e-1 with genuine
    # improvements as small as 1e-6 -- comparable to float32 rounding noise on
    # the score itself. A strict "<=" backtracking check reads that noise as
    # "no improving step at any scale" and stops after one iteration; a small
    # relative tolerance absorbs the noise without masking real increases.
    tol = 1e-5
    stall = 0
    for step in range(num_steps):
        grad, = torch.autograd.grad(loss, p)
        # The sum(p)=budget constraint only lets us move within the hyperplane
        # orthogonal to the all-ones vector, so subtract off the mean before
        # stepping: without this, a large uniform (DC) component of the
        # gradient -- present at any near-symmetric p, not only the uniform
        # init -- dominates the step-size selection below while contributing
        # almost nothing after re-projection, making backtracking declare
        # "converged" long before the informative part of the gradient has
        # been followed.
        grad = grad - grad.mean()
        # Backtracking: gradient magnitude varies by orders of magnitude across
        # this landscape (huge near the box boundary, small mid-interior), so a
        # fixed step size either stalls or overshoots into worse territory --
        # halve until the projected step actually decreases the objective (for a
        # non-convex objective this only guarantees a *non-increasing* sequence,
        # not global convergence, but that is exactly what a local search needs).
        accepted = False
        with torch.no_grad():
            for _ in range(30):
                candidate = project_capped_simplex(p - step_size * grad, budget)
                new_loss = value_at(candidate)
                if new_loss.item() <= loss.item() + tol * abs(loss.item()):
                    accepted = True
                    break
                step_size *= 0.5
            if accepted:
                p.copy_(candidate)
                # Relax back up, capped: without a ceiling, a run of accepted
                # steps (each multiplying by 1.5) grows step_size geometrically
                # forever, and once the true gradient is ~0 (e.g. this step
                # started from a stationary point) any leftover floating-point
                # noise in it gets amplified into a candidate far outside the
                # region the score function is numerically stable on.
                step_size = min(step_size * 1.5, max_step_size)
                stall = 0
            else:
                stall += 1
        loss = value_at(p)
        if verbose and (step % 50 == 0 or step == num_steps - 1):
            print(f'  {log_prefix}step={step:4d} {label}={loss.item():.6f} step_size={step_size:.2e}',
                  flush=True)
        # Patience: one noisy failed backtrack (stall=1) is not evidence of
        # convergence, only several in a row -- a real local optimum keeps
        # failing every time it's checked, transient noise does not.
        if stall >= 5:
            break

    with torch.no_grad():
        value = value_at(p).item()
        mask = torch.zeros(N, device=device)
        mask[torch.topk(p, budget).indices] = 1.0
    return {'p': p.detach(), 'value': value, 'mask': mask}


def _projected_gradient_search(
    oracle,
    budget: int,
    objective: str,
    num_steps: int,
    lr: float,
    eps: float,
    init: Optional[torch.Tensor],
    verbose: bool,
    log_prefix: str = '',
):
    """1D wrapper: routes :func:`_projected_gradient_core` through a :class:`LineOracle`."""
    def value_at(pp):
        return oracle.score(pp.clamp(eps, 1 - eps).unsqueeze(0), objective)[0]

    return _projected_gradient_core(value_at, oracle.W, budget, num_steps, lr, eps, init, verbose,
                                    oracle.device, log_prefix=log_prefix, label=objective)


def relaxed_design(
    oracle,
    budget: int,
    objective: str = 'logdet',
    num_steps: int = 300,
    lr: float = 0.5,
    eps: float = 1e-4,
    verbose: bool = True,
):
    """Solves the convex relaxation of D-/A-optimal line selection exactly.

    Both ``'logdet'`` and ``'trace'`` scores are convex in the sampling
    probabilities: each block ``G_a(p) = sum_k p_k (basis_k (o) R_a) + lam I``
    is an affine (in ``p``) map into the PSD cone, since ``basis_k`` is rank-1
    PSD and the Hadamard product of two PSD matrices is PSD (Schur product
    theorem); ``-logdet`` is convex on the PSD cone and ``tr(X^-1)`` is convex
    on it, and convexity survives composition with an affine map. So projected
    gradient descent on ``p in {0<=p<=1, sum p = budget}`` -- a convex set --
    converges to the *global* optimum of the relaxed problem, which is an upper
    bound on the true (binary-constrained) optimum: every 0/1 mask is also a
    feasible ``p``, so ``max over p'' >= max over binary masks >= greedy``. That
    makes the gap ``(relaxed - greedy) / relaxed`` a certificate on how far
    greedy can possibly be from optimal, with no brute force required.

    Not applicable to ``'mse'`` / ``'mse_noise'``: those score a specific
    realised image (via a bias term that is not convex in ``p``), not a convex
    information functional. See :func:`loupe_style_design` for those.

    Args:
        oracle: A :class:`LineOracle` or :class:`MeanLineOracle`.
        budget: Target number of sampled lines.
        objective: ``'logdet'`` or ``'trace'``.
        num_steps: Projected gradient steps.
        lr: Step size.
        eps: Keeps ``p`` strictly inside (0,1) so ``G(p)`` never touches
            singularity at exactly 0.
        verbose: Print progress.

    Returns:
        ``p`` [W] the relaxed optimum, ``value`` its score, and ``mask`` the
        top-``budget`` hard rounding of ``p``.
    """
    if objective not in ('logdet', 'trace'):
        raise ValueError("relaxed_design only certifies 'logdet' and 'trace'; "
                          "use loupe_style_design for 'mse'/'mse_noise'")
    return _projected_gradient_search(oracle, budget, objective, num_steps, lr, eps, None, verbose)


def loupe_style_design(
    oracle,
    budget: int,
    objective: str = 'mse_noise',
    num_steps: int = 300,
    lr: float = 0.5,
    eps: float = 1e-4,
    init: Optional[torch.Tensor] = None,
    verbose: bool = True,
    log_prefix: str = '',
):
    """Gradient descent on a relaxed sampling probability, for the same
    image-dependent objectives LOUPE trains against.

    This is exactly what a LOUPE mask-net converges to in the limit of a
    perfect (oracle) reconstructor and infinite sigmoid slope: a point ``p``
    on the capped simplex ``{0<=p<=1, sum p = budget}`` found by following the
    gradient of the reconstruction loss. Routing that search through
    :class:`LineOracle` instead of a trained UNet/VarNet removes every
    confound between "the sampler can't find a good mask" and "the
    reconstructor's gradient is too noisy/undertrained to point at one" --
    both share :func:`_projected_gradient_search` with :func:`relaxed_design`,
    so the only difference from that certified bound is that ``'mse'`` /
    ``'mse_noise'`` are not convex: the search still converges (to a
    stationary point), but with no guarantee it is the global one, and no
    guarantee it beats greedy. Whether it does is exactly the empirical
    question this function exists to answer.

    Args:
        oracle: A :class:`LineOracle` or :class:`MeanLineOracle`.
        budget: Target number of sampled lines.
        objective: ``'mse'`` or ``'mse_noise'``.
        num_steps: Projected gradient steps.
        lr: Step size.
        eps: Keeps ``p`` strictly inside (0,1).
        init: Optional [W] warm start for ``p`` (e.g. a greedy mask, to test
            whether the local search can hold or improve on it rather than
            find it from scratch).
        verbose: Print progress.
        log_prefix: Prepended to progress lines, useful when interleaving
            several runs' output.

    Returns:
        ``p`` [W] the local optimum, ``value`` its score, and ``mask`` the
        top-``budget`` hard rounding of ``p``.
    """
    if objective not in ('mse', 'mse_noise'):
        raise ValueError("loupe_style_design is for 'mse'/'mse_noise'; "
                          "use relaxed_design for the certified 'trace'/'logdet' bound")
    return _projected_gradient_search(oracle, budget, objective, num_steps, lr, eps, init, verbose,
                                      log_prefix=log_prefix)


# --------------------------------------------------------------------------
# 2D: matrix-free greedy
# --------------------------------------------------------------------------

def kspace_residual_scores(
    kspace: torch.Tensor,
    sense_maps: torch.Tensor,
    mask: torch.Tensor,
    recon: torch.Tensor,
) -> torch.Tensor:
    """Coil-summed squared k-space residual ``|y - A x|^2`` at every location.

    This is the 2D greedy criterion: one forward operator application scores all
    HW candidates simultaneously, since the residual at an unsampled location is
    exactly the data the current estimate fails to predict.

    Args:
        kspace: [C, H, W] complex fully sampled k-space.
        sense_maps: [C, H, W] complex coil sensitivities.
        mask: [H, W] current binary mask.
        recon: [H, W] complex current reconstruction.

    Returns:
        [H, W] real scores; already sampled locations are zeroed.
    """
    predicted = fft_2d_img(recon.unsqueeze(0) * sense_maps)
    return (kspace - predicted).abs().pow(2).sum(dim=0) * (1.0 - mask)


def estimate_leverage(
    sense_maps: torch.Tensor,
    mask: torch.Tensor,
    lam: float,
    num_probes: int = 8,
    cg_iters: int = 24,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Estimate ``a_k^H H^-1 a_k`` at every k-space location, matrix free.

    ``H = A^H A + lam I`` is HW x HW, so the per-candidate leverage that the
    exact greedy gain needs cannot be computed one candidate at a time. It can
    be sampled, though: if ``z`` is drawn with covariance ``H`` -- and it can be,
    because ``z = A^H n + sqrt(lam) w`` with white ``n``, ``w`` does exactly that
    -- then ``u = H^-1 z`` satisfies ``E[u u^H] = H^-1``. So

        d_k = a_k^H H^-1 a_k = E |(F S u)_k|^2

    and one CG solve plus one forward operator application scores every location
    at once. ``num_probes`` solves give an unbiased estimate of the whole map.

    Args:
        sense_maps: [C, H, W] complex coil sensitivities.
        mask: [H, W] current binary mask.
        lam: Tikhonov weight (absolute).
        num_probes: Probe vectors averaged.
        cg_iters: CG iterations per probe.
        generator: Optional RNG, for reproducible probes.

    Returns:
        [H, W] real leverage, summed over coils.
    """
    from activemri.reconstructors.CG_Sense import conjugate_gradient_sense

    C, H, W = sense_maps.shape
    shape = (num_probes, C, H, W)
    randn = lambda sz: torch.randn(*sz, 2, device=sense_maps.device, generator=generator)
    noise = torch.view_as_complex(randn(shape).contiguous()) / 2 ** 0.5
    white = torch.view_as_complex(randn((num_probes, H, W)).contiguous()) / 2 ** 0.5

    maps = sense_maps.unsqueeze(0).expand(num_probes, -1, -1, -1)
    m = mask.view(1, 1, H, W).expand(num_probes, -1, -1, -1)

    # z has covariance A^H A + lam I; recover it as the adjoint of masked noise.
    z = (ifft_2d_img(noise * m) * maps.conj()).sum(dim=1) + lam ** 0.5 * white
    u = conjugate_gradient_sense(
        fft_2d_img(z.unsqueeze(1) * maps) * m, maps, m, num_iters=cg_iters, lamda=lam,
        rhs=z,
    )
    return fft_2d_img(u.unsqueeze(1) * maps).abs().pow(2).sum(dim=1).mean(dim=0)


def dense_normal_operator(sense_maps: torch.Tensor, mask: torch.Tensor, lam: float,
                          chunk: int = 2048, out_device=None) -> torch.Tensor:
    """Materialises ``A^H A + lam I`` as a dense HW x HW matrix.

    Feasible only because HW is modest here (16384 at 128x128 -> 2.1GB, ~2s to
    build via batched forward-adjoint applications, ~2-3s to invert). This is
    the honest, non-probed alternative to :func:`estimate_leverage` -- it trades
    memory and O((HW)^2)-O((HW)^3) compute for an exact answer. At 256x256 the
    matrix alone is 34GB and inversion is ~8x slower per dimension cubed, so this
    stops being practical well before typical fastMRI image sizes.

    Args:
        sense_maps: [C, H, W] complex coil sensitivities.
        mask: [H, W] binary mask.
        lam: Tikhonov weight (absolute).
        chunk: Basis vectors applied per batch.
        out_device: Where the returned ``[HW, HW]`` matrix lives. Defaults to
            ``sense_maps.device``. Passing a device with more room than the GPU
            (e.g. ``'cpu'``) lets the *construction* still run at GPU speed --
            each chunk's FFTs happen on ``sense_maps.device`` and only that
            chunk (not the whole matrix) is ever resident on the compute
            device -- while the full matrix itself is only ever materialised on
            ``out_device``. Needed once ``(HW)^2 * 8`` bytes exceeds GPU memory,
            which happens well before it exceeds host RAM.

    Returns:
        [HW, HW] complex Hermitian positive definite matrix, on ``out_device``.
    """
    C, H, W = sense_maps.shape
    N = H * W
    out_device = out_device or sense_maps.device
    G = torch.empty(N, N, dtype=torch.complex64, device=out_device)
    for i in range(0, N, chunk):
        # Only this chunk's identity rows, not the full N x N identity -- at the
        # sizes this function is meant for, N x N already IS the whole memory
        # budget, so a second N x N tensor just to index into is wasteful.
        this_chunk = min(chunk, N - i)
        eye_chunk = torch.zeros(this_chunk, H, W, dtype=torch.complex64, device=sense_maps.device)
        eye_chunk.view(this_chunk, N)[torch.arange(this_chunk), torch.arange(i, i + this_chunk)] = 1.0
        coil = eye_chunk.unsqueeze(1) * sense_maps
        k = fft_2d_img(coil) * mask
        # A^H A applied to basis vector e_i is COLUMN i of the matrix, not row i:
        # G is Hermitian but not symmetric, so storing it as a row silently transposes
        # it into conj(G) instead of G.
        chunk_cols = (ifft_2d_img(k) * sense_maps.conj()).sum(dim=1).reshape(-1, N).T
        G[:, i:i + chunk] = chunk_cols.to(out_device)
    G.diagonal().add_(lam)  # avoid materialising another N x N identity just to add lam*I
    return G


def exact_leverage(sense_maps: torch.Tensor, mask: torch.Tensor, lam: float) -> torch.Tensor:
    """Exact ``a_k^H H^-1 a_k`` at every k-space location, via a dense inverse.

    ``a_k`` (a candidate row, ignoring the current mask) has coil-c entries
    ``F[k, :] * S_c``, so per coil the quadratic form is ``F Y_c F^H`` evaluated
    on the diagonal, with ``Y_c = diag(S_c) H^-1 diag(S_c)^*``. Both factors of
    ``F`` are applied as batched 2D transforms (right-multiplication by ``F^H``
    is an inverse FFT of every row of ``Y_c``; left-multiplication by ``F`` is a
    forward FFT of every column of the result), so no HW x HW matrix is
    Fourier-transformed densely -- only the diagonal is ever materialised.

    Args:
        sense_maps: [C, H, W] complex coil sensitivities.
        mask: [H, W] current binary mask.
        lam: Tikhonov weight (absolute).

    Returns:
        [H, W] real leverage, summed over coils.
    """
    C, H, W = sense_maps.shape
    N = H * W
    G = dense_normal_operator(sense_maps, mask, lam)
    G_inv = torch.linalg.solve(G, torch.eye(N, dtype=G.dtype, device=G.device))

    leverage = torch.zeros(N, dtype=torch.float32, device=sense_maps.device)
    s = sense_maps.reshape(C, N)
    for c in range(C):
        Y = s[c].unsqueeze(1) * G_inv * s[c].conj().unsqueeze(0)     # [N, N]
        Z = ifft_2d_img(Y.reshape(N, H, W)).reshape(N, N)             # Y @ F^H, per row
        R_diag = fft_2d_img(Z.T.reshape(N, H, W)).reshape(N, N).diagonal()  # (F Z)[k,k]
        leverage += R_diag.real
    return leverage.reshape(H, W)


def greedy_2d(
    kspace: torch.Tensor,
    sense_maps: torch.Tensor,
    budget: int,
    batch: int = 256,
    preselect: int = 0,
    cg_iters: int = 24,
    lam: float = 1e-2,
    score: str = 'residual',
    num_probes: int = 8,
    reference: Optional[torch.Tensor] = None,
    support: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> Dict:
    """Batched greedy selection of arbitrary 2D k-space locations.

    Selecting one point at a time is hopeless at HW candidates, so each round
    solves once, scores every candidate in a single forward pass, and commits
    the top ``batch`` points. Rounds cost ``budget / batch`` CG solves in total
    rather than one per candidate.

    Args:
        kspace: [C, H, W] complex fully sampled k-space.
        sense_maps: [C, H, W] complex coil sensitivities.
        budget: Total number of sampled k-space locations.
        batch: Points committed per round. Smaller is greedier and slower.
        preselect: Side length of a square centre block fixed up front.
        cg_iters: CG iterations per round.
        lam: Tikhonov weight.
        score: ``'residual'`` ranks by ``|y - A x|^2`` alone. ``'leverage'``
            divides by ``1 + a_k^H H^-1 a_k``, estimated by probing (cheap,
            noisy). ``'leverage_exact'`` computes the same quantity exactly via
            a dense HW x HW inverse -- affordable up to a few hundred squared
            pixels, not at typical fastMRI resolutions.
        num_probes: Probe vectors for the leverage estimate.
        reference: [H, W] complex fully sampled reconstruction, for reporting
            the error curve. Computed if omitted.
        support: [H, W] object mask, so the error curve is comparable with the
            1D oracle's.
        verbose: Print progress.

    Returns:
        ``mask`` [H, W] and the per-round normalised error ``history``.
    """
    from activemri.reconstructors.CG_Sense import conjugate_gradient_sense

    C, H, W = kspace.shape
    mask = torch.zeros(H, W, device=kspace.device)
    if preselect > 0:
        n = preselect // 2
        mask[H // 2 - n: H // 2 + n, W // 2 - n: W // 2 + n] = 1.0

    def solve(m: torch.Tensor) -> torch.Tensor:
        return conjugate_gradient_sense(
            (kspace * m).unsqueeze(0), sense_maps.unsqueeze(0), m.view(1, 1, H, W),
            num_iters=cg_iters, lamda=lam,
        )[0]

    if support is None:
        support = torch.ones(H, W, device=kspace.device)
    if reference is None:
        reference = solve(torch.ones_like(mask))
    reference = reference * support
    ref_energy = reference.abs().pow(2).sum().clamp(min=1e-12)

    history = []
    while True:
        recon = solve(mask) if mask.any() else torch.zeros(H, W, dtype=kspace.dtype, device=kspace.device)
        history.append((((recon - reference).abs().pow(2) * support).sum() / ref_energy).item())
        if verbose:
            print(f'  points={int(mask.sum().item()):6d}  nmse={history[-1]:.6g}', flush=True)

        remaining = budget - int(mask.sum().item())
        if remaining <= 0:
            break

        scores = kspace_residual_scores(kspace, sense_maps, mask, recon)
        if score == 'leverage':
            leverage = estimate_leverage(sense_maps, mask, lam, num_probes, cg_iters)
            scores = scores / (1.0 + leverage)
        elif score == 'leverage_exact':
            leverage = exact_leverage(sense_maps, mask, lam)
            scores = scores / (1.0 + leverage)
        take = min(batch, remaining)
        idx = torch.topk(scores.flatten(), take).indices
        mask.view(-1)[idx] = 1.0

    return {'mask': mask, 'history': history}


def _cg_solve_batch(kspace, sense_maps, masks, cg_iters, lam):
    """Batched CG-SENSE reconstruction for a stack of candidate masks.

    Args:
        kspace: [C, H, W] complex, one slice's fully sampled k-space.
        sense_maps: [C, H, W] complex coil sensitivities.
        masks: [B, H, W] real, possibly soft (0..1) sampling weights.
        cg_iters: CG iterations.
        lam: Tikhonov weight.

    Returns:
        [B, H, W] complex reconstructions.
    """
    from activemri.reconstructors.CG_Sense import conjugate_gradient_sense

    B = masks.shape[0]
    m = masks.unsqueeze(1)  # [B, 1, H, W]
    k = kspace.unsqueeze(0) * m
    s = sense_maps.unsqueeze(0).expand(B, -1, -1, -1)
    return conjugate_gradient_sense(k, s, m, num_iters=cg_iters, lamda=lam)


def loupe_style_design_2d(
    kspace: torch.Tensor,
    sense_maps: torch.Tensor,
    budget: int,
    cg_iters: int = 10,
    lam: float = 1e-2,
    num_steps: int = 100,
    lr: float = 0.5,
    eps: float = 1e-4,
    init: Optional[torch.Tensor] = None,
    reference: Optional[torch.Tensor] = None,
    support: Optional[torch.Tensor] = None,
    verbose: bool = True,
    log_prefix: str = '',
) -> Dict:
    """2D analogue of :func:`loupe_style_design`: gradient descent on a soft
    2D sampling density, through a *differentiable* CG-SENSE solve.

    There is no closed-form oracle in 2D (see the module docstring), so this
    routes :func:`_projected_gradient_core` through
    :func:`conjugate_gradient_sense` directly -- the same reconstructor
    :func:`greedy_2d` uses, just called with gradients enabled instead of
    inside ``torch.no_grad()``. This is exactly the object a real LOUPE
    ``ProbMask`` converges to given a perfect (oracle) reconstructor and
    infinite sigmoid slope, with every confound from an imperfect or
    undertrained neural reconstructor removed.

    Args:
        kspace: [C, H, W] complex fully sampled k-space.
        sense_maps: [C, H, W] complex coil sensitivities.
        budget: Total number of sampled k-space locations.
        cg_iters: CG iterations per solve (backpropagated through, so this
            trades wall-clock and graph depth for reconstruction accuracy).
        lam: Tikhonov weight.
        num_steps: Projected gradient steps.
        lr: Step size.
        eps: Keeps ``p`` strictly inside (0,1).
        init: Optional [H, W] warm start for ``p`` (e.g. a greedy_2d mask).
        reference: [H, W] complex fully sampled reconstruction; computed if omitted.
        support: [H, W] object mask.
        verbose: Print progress.
        log_prefix: Prepended to progress lines.

    Returns:
        ``p`` [H, W] the local optimum, ``value`` its NMSE, and ``mask`` the
        top-``budget`` hard rounding of ``p``.
    """
    C, H, W = kspace.shape
    N = H * W
    device = kspace.device

    if support is None:
        support = torch.ones(H, W, device=device)
    if reference is None:
        with torch.no_grad():
            reference = _cg_solve_batch(kspace, sense_maps, torch.ones(1, H, W, device=device),
                                        cg_iters, lam)[0]
    reference = (reference * support).detach()
    ref_energy = reference.abs().pow(2).sum().clamp(min=1e-12)

    def value_at(p_flat):
        recon = _cg_solve_batch(kspace, sense_maps, p_flat.view(1, H, W), cg_iters, lam)[0]
        return ((recon - reference).abs().pow(2) * support).sum() / ref_energy

    init_flat = init.reshape(-1) if init is not None else None
    result = _projected_gradient_core(value_at, N, budget, num_steps, lr, eps, init_flat, verbose,
                                      device, log_prefix=log_prefix, label='nmse')
    return {'p': result['p'].view(H, W), 'value': result['value'], 'mask': result['mask'].view(H, W)}


def reinforce_2d(
    kspace: torch.Tensor,
    sense_maps: torch.Tensor,
    budget: int,
    batch: int = 128,
    preselect: int = 0,
    cg_iters: int = 10,
    lam: float = 1e-2,
    episodes_per_update: int = 6,
    num_updates: int = 20,
    lr: float = 0.2,
    baseline_momentum: float = 0.9,
    init_logits: Optional[torch.Tensor] = None,
    init_temperature: float = 2.0,
    reference: Optional[torch.Tensor] = None,
    support: Optional[torch.Tensor] = None,
    verbose: bool = True,
    log_prefix: str = '',
    entropy_weight: float = 0.0,
    clip_grad_norm: Optional[float] = None,
    normalize_advantages: bool = False,
) -> Dict:
    """Batched macro-round REINFORCE over 2D k-space locations.

    A whole-mask Plackett-Luce policy like :func:`reinforce_lines` does not
    scale to 2D: its log-probability is an ``O(budget)`` Python loop, and
    unlike the closed-form :class:`LineOracle`, every 2D reward needs an
    actual CG solve -- there is no cheap way to score a big batch of full
    masks per episode. This instead mirrors :func:`greedy_2d`'s own round
    structure (commit ``batch`` points per round, one CG solve per round) and
    makes *that* round's batch choice the learned, stochastic part:

      - the Plackett-Luce draw each round only has to rank ``batch``
        candidates out of what's left, not the whole ``budget`` -- so its
        log-probability is ``O(batch)``, not ``O(budget)``;
      - several independent rollouts (``episodes_per_update``) share one
        logit map ``theta`` over all H*W locations and are advanced through
        one round together via a single *batched* CG solve, so one
        policy-gradient update costs ``num_rounds`` batched solves -- the
        same order as one call to :func:`greedy_2d` itself, not
        ``num_rounds`` times more.

    Credit is assigned per round via reward-to-go (the sum of this round's
    and every later round's NMSE improvement), each with its own EMA
    baseline, rather than crediting every round with the whole trajectory's
    return -- standard REINFORCE variance reduction, and cheap here since
    the number of rounds is small (``budget / batch``).

    Args:
        kspace: [C, H, W] complex fully sampled k-space.
        sense_maps: [C, H, W] complex coil sensitivities.
        budget: Total number of sampled k-space locations.
        batch: Points committed per round (== that round's Plackett-Luce k).
        preselect: Side length of a square centre block fixed up front,
            identically for every rollout (matches :func:`greedy_2d`).
        cg_iters: CG iterations per solve.
        lam: Tikhonov weight.
        episodes_per_update: Parallel stochastic rollouts sharing one batched
            CG solve per round, averaged into one policy-gradient update.
        num_updates: Policy-gradient updates.
        lr: Adam learning rate on the logits.
        baseline_momentum: EMA decay for each round's REINFORCE baseline.
        init_logits: Optional [H, W] prior score field to start ``theta``
            from (e.g. :func:`kspace_residual_scores` or
            :func:`estimate_leverage` against the preselected mask alone) --
            standardised and scaled by ``init_temperature`` rather than used
            verbatim, so the policy starts *biased toward* that prior instead
            of deterministically reproducing it (which would leave nothing
            for the policy gradient to explore). Defaults to zero (uniform),
            i.e. no prior.
        init_temperature: Scale applied to the standardised ``init_logits``.
            Larger values start closer to argmax-ing the prior; 0 is
            equivalent to no prior.
        reference: [H, W] complex fully sampled reconstruction; computed if omitted.
        support: [H, W] object mask.
        verbose: Print progress.
        entropy_weight: Coefficient for entropy regularization. Larger values
            encourage exploration. Default 0 (no regularization); try 0.01-0.1
            for improved convergence stability.
        clip_grad_norm: If set, gradient norm is clipped to this value before
            optimizer step. Prevents large gradient spikes. Default None (no clipping);
            try 1.0 for better stability.
        normalize_advantages: If True, advantages are normalized per round
            (zero-meaned and std-normalized). Reduces impact of scale differences
            across rounds. Default False for backward compatibility.

    Returns:
        ``mask`` [H, W] (the deterministic top-batch-per-round rollout under
        the learned logits), ``logits`` [H, W], and the per-update mean final
        NMSE ``history``.
    """
    C, H, W = kspace.shape
    N = H * W
    device = kspace.device

    preselect_mask = torch.zeros(H, W, device=device)
    if preselect > 0:
        n = preselect // 2
        preselect_mask[H // 2 - n: H // 2 + n, W // 2 - n: W // 2 + n] = 1.0
    num_preselect = int(preselect_mask.sum().item())
    if budget < num_preselect:
        raise ValueError('budget must be >= preselect^2')

    if support is None:
        support = torch.ones(H, W, device=device)
    if reference is None:
        with torch.no_grad():
            reference = _cg_solve_batch(kspace, sense_maps, torch.ones(1, H, W, device=device),
                                        cg_iters, lam)[0]
    reference = (reference * support).detach()
    ref_energy = reference.abs().pow(2).sum().clamp(min=1e-12)

    def nmse(recon):  # [B, H, W] -> [B]
        return ((recon - reference).abs().pow(2) * support).sum(dim=(-2, -1)) / ref_energy

    rounds = []
    remaining = budget - num_preselect
    while remaining > 0:
        rounds.append(min(batch, remaining))
        remaining -= rounds[-1]

    if init_logits is None:
        theta_init = torch.zeros(N, device=device)
    else:
        flat = init_logits.reshape(-1).to(device)
        # Standardise before scaling: init_logits' raw units (squared k-space
        # residual, or a leverage estimate) are meaningless to a Plackett-Luce
        # softmax on their own -- only relative, unit-scaled differences
        # matter, and init_temperature is what controls how sharp the prior
        # is on top of that.
        theta_init = (flat - flat.mean()) / flat.std().clamp(min=1e-12) * init_temperature
    theta = theta_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    baselines = [None] * len(rounds)
    history = []

    for update in range(num_updates):
        E = episodes_per_update
        masks = preselect_mask.view(-1).unsqueeze(0).repeat(E, 1)  # [E, N]
        taken = masks.bool()
        with torch.no_grad():
            recon = (_cg_solve_batch(kspace, sense_maps, masks.view(E, H, W), cg_iters, lam)
                     if num_preselect > 0 else torch.zeros(E, H, W, dtype=kspace.dtype, device=device))
            prev_err = nmse(recon)

        round_logps, round_rewards = [], []
        for take in rounds:
            logits = theta.unsqueeze(0).expand(E, -1).masked_fill(taken, float('-inf'))
            order, logp = plackett_luce_sample_and_logprob(logits, take)
            round_logps.append(logp)

            # Out-of-place, as in plackett_luce_sample_and_logprob: the graph
            # needs `taken`/`masks` as they were at each step for backward.
            masks = masks.scatter(1, order, 1.0)
            taken = taken | torch.zeros_like(taken).scatter(1, order, True)

            with torch.no_grad():
                recon = _cg_solve_batch(kspace, sense_maps, masks.view(E, H, W), cg_iters, lam)
                err = nmse(recon)
            round_rewards.append((prev_err - err).detach())  # this round's NMSE improvement
            prev_err = err

        # Reward-to-go per round, each against its own running baseline.
        returns = [None] * len(rounds)
        running = torch.zeros(E, device=device)
        for t in reversed(range(len(rounds))):
            running = running + round_rewards[t]
            returns[t] = running.clone()

        loss = 0.0
        for t in range(len(rounds)):
            baselines[t] = returns[t].mean().detach() if baselines[t] is None else (
                baseline_momentum * baselines[t] + (1 - baseline_momentum) * returns[t].mean().detach())
            advantage = returns[t] - baselines[t]
            if normalize_advantages:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            loss = loss - (advantage.detach() * round_logps[t]).mean()

        if entropy_weight > 0:
            logits = theta.unsqueeze(0).expand(E, -1).masked_fill(~preselect_mask.view(-1).bool().unsqueeze(0), float('-inf'))
            probs = torch.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            loss = loss - entropy_weight * entropy

        optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_([theta], max_norm=clip_grad_norm)
        optimizer.step()

        history.append(prev_err.mean().item())
        if verbose:
            print(f'  {log_prefix}update={update:4d}  mean_final_nmse={history[-1]:.6g}  '
                  f'best_in_batch={prev_err.min().item():.6g}', flush=True)

    # Deterministic rollout under the learned logits: top-`take` of what's
    # left each round, no sampling -- the mask actually handed back for use.
    with torch.no_grad():
        mask = preselect_mask.view(-1).clone()
        taken = mask.bool()
        for take in rounds:
            logits = theta.masked_fill(taken, float('-inf'))
            idx = torch.topk(logits, take).indices
            mask[idx] = 1.0
            taken = taken | torch.zeros_like(taken).scatter(0, idx, True)

    return {'mask': mask.view(H, W), 'logits': theta.detach().view(H, W), 'history': history}
