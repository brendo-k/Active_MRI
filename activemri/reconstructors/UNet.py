"""
Portion of this code is from fastmri(https://github.com/facebookresearch/fastMRI)

Copyright (c) Facebook, Inc. and its affiliates.

Licensed under the MIT License.
"""

import torch
from torch import nn
from torch.nn import functional as F
from activemri.baselines.loupe_codes.transforms import *
from activemri.reconstructors.LearnedSampling import *
from activemri.utils.image_processing import ifft_2d_img, fft_2d_img



class LOUPEUNet(nn.Module):
    """
        PyTorch implementation of a U-Net model.

        The model takes a real or complex value image and use a UNet to denoise the image.
        A residual connection is applied to stablize training.
    """
    def __init__(self,
                 in_chans,
                 out_chans,
                 chans,
                 depth,
                 drop_prob,
                 with_uncertainty=False):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            depth (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.with_uncertainty = with_uncertainty

        if with_uncertainty:
            out_chans = out_chans + 1

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.depth = depth
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([downsample_block(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(depth - 1):
            self.down_sample_layers.append(downsample_block(ch, ch * 2, drop_prob))
            ch *= 2

        self.bottleneck = ConvBlock(ch, ch * 2, drop_prob)
        ch *= 2

        self.up_sample_layers = nn.ModuleList()
        for _ in range(depth):
            self.up_sample_layers.append(upsample_block(ch, ch // 2, drop_prob))
            ch //= 2

        self.final_conv = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, input):
        skips = []
        x = input

        for down in self.down_sample_layers:
            x, skip = down(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for up, skip in zip(self.up_sample_layers, reversed(skips)):
            x = up(x, skip)

        x = self.final_conv(x)

        if self.with_uncertainty:
            x[:, :self.in_chans] = x[:, :self.in_chans] + input
        else:
            x = x + input[:, :self.out_chans]

        return x
    
class upsample_block(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.layers = ConvBlock(in_chans, out_chans, drop_prob)

    def forward(self, input, skip_features):
        input = self.upsample(input)
        input = torch.cat([input, skip_features], dim=1)
        return self.layers(input)
    
    
class downsample_block(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()
        self.layers = ConvBlock(in_chans, out_chans, drop_prob)

    def forward(self, input):
        output = self.layers(input)
        output_downsampled = F.max_pool2d(output, kernel_size=2)
        return output_downsampled, output

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """
    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super(ConvBlock, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'