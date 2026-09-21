import torch 
import torch.nn as nn
import torch.nn.functional as F
from activemri.utils.image_processing import ifft_2d_img, fft_2d_img

def _sense_normal(x: torch.Tensor, sense_maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Applies A^H A to x, where A = mask * F * diag(sense_maps).

    Args:
        x: Image estimate [B, H, W] complex.
        sense_maps: Coil sensitivity maps [B, C, H, W] complex.
        mask: Binary sampling mask [B, 1, H, W].

    Returns:
        A^H A x with shape [B, H, W] complex.
    """
    # A: expand to coil images, FFT, apply mask
    coil_imgs = x.unsqueeze(1) * sense_maps          # [B, C, H, W]
    coil_kspace = fft_2d_img(coil_imgs)               # [B, C, H, W]
    masked_kspace = coil_kspace * mask                # [B, C, H, W]  (mask broadcasts over C)
    # A^H: IFFT, weight by conj sensitivity maps, sum over coils
    coil_imgs_back = ifft_2d_img(masked_kspace)       # [B, C, H, W]
    return (coil_imgs_back * sense_maps.conj()).sum(dim=1)  # [B, H, W]


def _sense_adjoint(kspace: torch.Tensor, sense_maps: torch.Tensor) -> torch.Tensor:
    """Applies A^H to observed k-space: A^H y = sum_c S_c^H F^H y_c.

    Args:
        kspace: Masked multi-coil k-space [B, C, H, W] complex.
        sense_maps: Coil sensitivity maps [B, C, H, W] complex.

    Returns:
        Combined image [B, H, W] complex.
    """
    coil_imgs = ifft_2d_img(kspace)                   # [B, C, H, W]
    return (coil_imgs * sense_maps.conj()).sum(dim=1)  # [B, H, W]


def conjugate_gradient_sense(
    masked_kspace: torch.Tensor,
    sense_maps: torch.Tensor,
    mask: torch.Tensor,
    num_iters: int = 10,
    lamda: float = 0.0,
    rhs: torch.Tensor = None,
) -> torch.Tensor:
    """Solves the SENSE reconstruction via conjugate gradient.

    Minimises ||A x - y||^2 + lamda ||x||^2, i.e. solves
    (A^H A + lamda I) x = A^H y.

    Args:
        masked_kspace: Observed multi-coil k-space [B, C, H, W] complex.
        sense_maps: Coil sensitivity maps [B, C, H, W] complex.
        mask: Binary sampling mask [B, 1, H, W].
        num_iters: Number of CG iterations.
        lamda: Tikhonov regularisation weight.
        rhs: Optional right-hand side [B, H, W] complex, used instead of
            ``A^H y``. Lets callers solve against a vector that is not the
            adjoint of any k-space data, as the leverage probes do.

    Returns:
        Reconstructed image [B, H, W] complex.
    """
    b = _sense_adjoint(masked_kspace, sense_maps) if rhs is None else rhs  # [B, H, W]

    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rTr = (r.conj() * r).real.sum(dim=(-2, -1), keepdim=True)  # [B, 1, 1]

    for _ in range(num_iters):
        Ap = _sense_normal(p, sense_maps, mask)
        if lamda > 0.0:
            Ap = Ap + lamda * p

        pTAp = (p.conj() * Ap).real.sum(dim=(-2, -1), keepdim=True)  # [B, 1, 1]
        alpha = rTr / pTAp.clamp(min=1e-12)

        x = x + alpha * p
        r = r - alpha * Ap

        rTr_new = (r.conj() * r).real.sum(dim=(-2, -1), keepdim=True)
        beta = rTr_new / rTr.clamp(min=1e-12)
        p = r + beta * p
        rTr = rTr_new

    return x


def compute_gfactor(
    sense_maps: torch.Tensor,
    mask: torch.Tensor,
    num_replicas: int = 4,
    cg_iters: int = 5,
) -> torch.Tensor:
    """Compute a g-factor map via the batched pseudo-replica method.

    Generates all noise replicas at once and reconstructs them in a single
    batched CG-SENSE call for efficiency. Each sample in the input batch is
    processed independently.

    Args:
        sense_maps: Coil sensitivity maps [B, C, H, W] complex.
        mask: Binary sampling mask [B, 1, H, W].
        num_replicas: Number of noise replicas (more = smoother, but 4 is fast).
        cg_iters: CG iterations per replica (5 is sufficient for noise propagation).

    Returns:
        g-factor map [B, 1, H, W] real (>=1 inside object).
    """
    B, C, H, W = sense_maps.shape
    g_maps = []

    for b in range(B):
        s = sense_maps[b:b+1].expand(num_replicas, -1, -1, -1)  # [N, C, H, W]
        m = mask[b:b+1].expand(num_replicas, -1, -1, -1)        # [N, 1, H, W]

        noise_k = torch.randn(num_replicas, C, H, W, 2, device=sense_maps.device) * (1.0 / (H * W) ** 0.5)
        noise_k = torch.view_as_complex(noise_k.contiguous()) * m

        recons = conjugate_gradient_sense(noise_k, s, m, num_iters=cg_iters, lamda=0.0)  # [N, H, W]
        noise_std = recons.real.std(dim=0)  # [H, W]

        rss_sense = sense_maps[b].abs().pow(2).sum(dim=0).sqrt().clamp(min=1e-8)
        sigma_input = 1.0 / (H * W) ** 0.5
        sigma_baseline = sigma_input / rss_sense

        g_maps.append(noise_std / sigma_baseline.clamp(min=1e-8))

    return torch.stack(g_maps, dim=0).unsqueeze(1)  # [B, 1, H, W]


class CGSENSEReconstructor(nn.Module):
    """Wraps :func:`conjugate_gradient_sense` as a drop-in ``nn.Module``.

    Has no learnable parameters; reconstruction is purely data-driven.
    """

    def __init__(self, num_iters: int = 10, lamda: float = 0.0):
        super().__init__()
        self.num_iters = num_iters
        self.lamda = lamda

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sense_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            masked_kspace: [B, C, H, W] complex.
            sense_maps: [B, C, H, W] complex.
            mask: [B, 1, H, W] binary float.

        Returns:
            Reconstructed image [B, 1, H, W] real (magnitude).
        """
        x = conjugate_gradient_sense(
            masked_kspace, sense_maps, mask, self.num_iters, self.lamda
        )
        return x.abs().unsqueeze(1)  # [B, 1, H, W]