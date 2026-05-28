"""
Portion of this code is from fastmri(https://github.com/facebookresearch/fastMRI)

Copyright (c) Facebook, Inc. and its affiliates.

Licensed under the MIT License.
"""

import torch
from torch import nn
from torch.nn import functional as F
from activemri.baselines.loupe_codes.transforms import *
from activemri.baselines.loupe_codes.layers import *
from activemri.baselines.sequential_sampling_codes.image_processing import ifft_2d_img, fft_2d_img


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

class LOUPEUNet(nn.Module):
    """
        PyTorch implementation of a U-Net model.
        This is based on:
            Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
            for biomedical image segmentation. In International Conference on Medical image
            computing and computer-assisted intervention, pages 234–241. Springer, 2015.   

        The model takes a real or complex value image and use a UNet to denoise the image. 
        A residual connection is applied to stablize training.       
    """
    def __init__(self,
                 in_chans,
                 out_chans,
                 chans,
                 num_pool_layers,
                 drop_prob,
#                  mask_length,
                 bi_dir=False,
                 old_recon=False,
                 with_uncertainty=False):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.old_recon = old_recon 
        if old_recon:
            assert 0 
            in_chans = in_chans+1 # add mask dim and old reconstruction dim  

        self.with_uncertainty = with_uncertainty

        if with_uncertainty:
            out_chans = out_chans+1

        self.in_chans = in_chans 
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

        nn.init.normal_(self.conv2[-1].weight, mean=0, std=0.001) 
        self.conv2[-1].bias.data.fill_(0)

    def forward(self, input, old_recon=None, eps=1e-8):
        # input: NCHW 
        # output: NCHW 

        if self.old_recon:
            assert 0 
            output = torch.cat([input, old_recon], dim=1)
        else:
            output = input 
        
        stack = []
        # print(input.shape, mask.shape)

        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)

        out_conv2 = self.conv2(output)

        b, c, h, w = out_conv2.shape
        if self.with_uncertainty:
            img_residual = out_conv2[:, :c-1, :, :]
            uncertainty = out_conv2[:, -1, :, :].unsqueeze(1)
        else:
            img_residual = out_conv2
            uncertainty = None

        if self.old_recon:
            return img_residual + old_recon
        else:
            return img_residual + input