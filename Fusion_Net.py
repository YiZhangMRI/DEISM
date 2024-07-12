"""
modified by Jianping Xu, 2/17/2023

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F


class Fusion_Net(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers_1 = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        self.down_sample_layers_2 = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers-1):
            ch *= 2
            self.down_sample_layers_1.append(ConvBlock(ch, ch, drop_prob))
            self.down_sample_layers_2.append(ConvBlock(ch, ch, drop_prob))

        self.conv = ConvBlock(2*ch, 2*ch, drop_prob) # bottom
        # ch //= 2
        # self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        # self.up_transpose_conv = nn.ModuleList()
        self.upsampling = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            # self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.upsampling.append(UndersamplingBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 3, ch, drop_prob))
            ch //= 2

        # self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.upsampling.append(UndersamplingBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 3, ch, drop_prob),
                # ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image_1, image_2):
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack_1 = []
        stack_2 = []
        output_1 = image_1
        output_2 = image_2

        # apply down-sampling layers
        for layer_1, layer_2 in zip(self.down_sample_layers_1, self.down_sample_layers_2):
            output_1 = layer_1(output_1) # in->ch->2ch->4ch->8ch
            stack_1.append(output_1)
            output_1_tmp = F.avg_pool2d(output_1, kernel_size=2, stride=2, padding=0)

            output_2 = layer_2(output_2)
            stack_2.append(output_2)
            output_2_tmp = F.avg_pool2d(output_2, kernel_size=2, stride=2, padding=0)

            output_1 = torch.cat([output_1_tmp, output_2_tmp], dim=1)
            output_2 = torch.cat([output_2_tmp, output_1_tmp], dim=1)

        output = output_1
        output = self.conv(output) # [B 256 6 6]

        # apply up-sampling layers
        # for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
        for upsampling, conv in zip(self.upsampling, self.up_conv):
            downsample_layer_1 = stack_1.pop()
            downsample_layer_2 = stack_2.pop()
            output = upsampling(output)
            # output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer_1.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer_1.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer_1, downsample_layer_2], dim=1)
            output = conv(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            #nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            #nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)


class UndersamplingBlock(nn.Module):
    """
    A Undersampling Block that consists of one convolution layers
    followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_chans, out_chans, kernel_size=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
