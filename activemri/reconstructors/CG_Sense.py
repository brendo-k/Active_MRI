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

    Returns:
        Reconstructed image [B, H, W] complex.
    """
    b = _sense_adjoint(masked_kspace, sense_maps)  # [B, H, W]

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