import torch
from torch import nn
from torch.nn import functional as F
from activemri.reconstructors.UNet import LOUPEUNet


class SampProb2D(nn.Module):
    """
        A 2D sampling probability model based on U-Net. Given current recon and previous mask, generate a new sampling probability in 2D.
    """
    def __init__(self, fixed_input=False):
        super().__init__()

        self.unet = LOUPEUNet(
            in_chans=5,
            out_chans=1,
            chans=64,
            depth=4,
            drop_prob=0,
        )

        self.fixed_input = fixed_input
        if fixed_input:
            fixed_input_tensor = torch.randn(size=[1, 5, 128, 128])
            self.fixed_input_tensor = nn.Parameter(fixed_input_tensor, requires_grad=False)

    def forward(self, x, mask):
        """
            Takes observed kspace data and sampling trajectories as input
            and outputs next sampling probability mask. Previously sampled
            locations are masked out.
            input: NHWC
            mask:  N1HW
        """
        x = x.permute(0, -1, 1, 2)
        x = torch.cat([x, mask], dim=1)

        if self.fixed_input:
            x = self.fixed_input_tensor

        out = self.unet(x)
        out = F.softplus(out)

        max_val = out.flatten(1).max(dim=1)[0].reshape(-1, 1, 1, 1)
        out = out / max_val

        return out * (1 - mask)
