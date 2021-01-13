import math

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.prelu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        return x + y


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    def __init__(self, n_residual_blocks: int, upsample_factor: int, base_filters=64):
        super().__init__()

        self.residual_blocks_num = n_residual_blocks
        self.base_filter = base_filters
        self.upsample_block_num = int(math.log(upsample_factor, 2))

        self.first_block = nn.Sequential(
            nn.Conv2d(3, base_filters, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.residual_blocks = nn.ModuleList([ResidualBlock(base_filters) for _ in range(n_residual_blocks)])

        self.post_residual_blocks = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters)
        )

        self.upsample_blocks = nn.ModuleList([UpsampleBlock(base_filters, 2) for _ in range(self.upsample_block_num)])

        self.last_conv = nn.Conv2d(base_filters, 3, kernel_size=9, padding=4)

    def forward(self, x):
        first_output = self.first_block(x)

        output = first_output.clone()
        for module in self.residual_blocks:
            output = module(output)

        output = self.post_residual_blocks(output) + first_output

        for module in self.upsample_blocks:
            output = module(output)

        return F.tanh(self.last_conv(output))