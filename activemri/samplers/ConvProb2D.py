import torch
from torch import nn
from torch.nn import functional as F
from activemri.reconstructors.UNet import LOUPEUNet, ConvBlock, upsample_block


class SampProb2D(nn.Module):
    """2D sampling probability model based on U-Net.

    Given current recon, k-space observations, and previous mask, generates
    a new sampling probability in 2D. Optionally outputs an uncertainty map.

    Args:
        fixed_input: If True, use a fixed learnable input tensor (debugging).
        with_recon: If True, expect reconstructed image (2ch complex) concatenated
            to the input, increasing in_chans by 2.
        with_gfactor: If True, expect a g-factor map (1ch) concatenated to the
            input, increasing in_chans by 1.
        uncertainty_mode: One of 'none', 'channel', or 'head'.
            - 'none': No uncertainty output.
            - 'channel': Extra output channel on the same UNet (out_chans=2).
            - 'head': Separate decoder head sharing the encoder.
    """

    def __init__(self, fixed_input=False, with_recon=False, with_gfactor=False, uncertainty_mode='none'):
        super().__init__()

        self.with_recon = with_recon
        self.with_gfactor = with_gfactor
        self.uncertainty_mode = uncertainty_mode
        in_chans = 5 + (2 if with_recon else 0) + (1 if with_gfactor else 0)
        out_chans = 2 if uncertainty_mode == 'channel' else 1

        self.unet = LOUPEUNet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=64,
            depth=4,
            drop_prob=0,
            upsample_mode='bilinear',
        )

        if uncertainty_mode == 'head':
            self.uncertainty_decoder = nn.ModuleList()
            ch = 64 * (2 ** 4)  # bottleneck channels = 1024
            for _ in range(4):
                self.uncertainty_decoder.append(upsample_block(ch, ch // 2, drop_prob=0))
                ch //= 2
            self.uncertainty_conv = nn.Conv2d(ch, 1, kernel_size=1)

        self.fixed_input = fixed_input
        if fixed_input:
            fixed_input_tensor = torch.randn(size=[1, in_chans, 128, 128])
            self.fixed_input_tensor = nn.Parameter(fixed_input_tensor, requires_grad=False)

    def forward(self, x, mask, recon=None, gfactor=None):
        """
        Args:
            x: Observed kspace data, shape NHWC (4 channels: full + masked, real+imag)
            mask: Previous sampling mask, shape N1HW
            recon: Optional reconstructed image, shape [N, 2, H, W] (real+imag)
            gfactor: Optional g-factor map, shape [N, 1, H, W]

        Returns:
            prob_mask: Sampling probability [N, 1, H, W]
            uncertainty: Uncertainty map [N, 1, H, W] or None
        """
        x = x.permute(0, -1, 1, 2)  # [N, 4, H, W]

        channels = [x]
        if self.with_recon:
            if recon is None:
                raise ValueError("with_recon=True but no recon provided")
            channels.append(recon)
        if self.with_gfactor:
            if gfactor is None:
                gfactor = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
            channels.append(gfactor)
        channels.append(mask)
        x = torch.cat(channels, dim=1)

        if self.fixed_input:
            x = self.fixed_input_tensor

        if self.uncertainty_mode == 'head':
            out, skips, bottleneck = self._forward_encoder(x)
            prob = F.softplus(out[:, :1])

            unc_x = bottleneck
            for up, skip in zip(self.uncertainty_decoder, reversed(skips)):
                unc_x = up(unc_x, skip)
            uncertainty = self.uncertainty_conv(unc_x)
        else:
            out = self.unet(x)

            if self.uncertainty_mode == 'channel':
                prob = F.softplus(out[:, :1])
                uncertainty = out[:, 1:2]
            else:
                prob = F.softplus(out)
                uncertainty = None

        max_val = prob.flatten(1).max(dim=1)[0].reshape(-1, 1, 1, 1)
        prob = prob / max_val

        prob = prob * (1 - mask)

        return prob, uncertainty

    def _forward_encoder(self, x):
        """Run the shared encoder and return prob output + intermediate states for the uncertainty head."""
        skips = []
        enc_x = x
        for down in self.unet.down_sample_layers:
            enc_x, skip = down(enc_x)
            skips.append(skip)

        bottleneck = self.unet.bottleneck(enc_x)

        dec_x = bottleneck
        for up, skip in zip(self.unet.up_sample_layers, reversed(skips)):
            dec_x = up(dec_x, skip)

        out = self.unet.final_conv(dec_x)
        out = out + x[:, :self.unet.out_chans]

        return out, skips, bottleneck
