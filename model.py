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

class Discriminator(nn.Module):
    def __init__(self, base_filters=64):
        super().__init__()

        self.conv1 = nn.Conv2d(3, base_filters, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(base_filters, base_filters, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters)

        self.conv3 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filters * 2)

        self.conv4 = nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filters * 2)

        self.conv5 = nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(base_filters * 4)

        self.conv6 = nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(base_filters * 4)

        self.conv7 = nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(base_filters * 4)

        self.conv8 = nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(base_filters * 8)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.post_pooling_conv = nn.Conv2d(base_filters * 8, base_filters * 16, kernel_size=1)
        self.output_conv = nn.Conv2d(base_filters * 16, 1, kernel_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        output = F.leaky_relu(self.conv1(x), 0.2)
        output = F.leaky_relu(self.bn2(self.conv2(output)), 0.2)
        output = F.leaky_relu(self.bn3(self.conv3(output)), 0.2)
        output = F.leaky_relu(self.bn4(self.conv4(output)), 0.2)
        output = F.leaky_relu(self.bn5(self.conv5(output)), 0.2)
        output = F.leaky_relu(self.bn6(self.conv6(output)), 0.2)
        output = F.leaky_relu(self.bn7(self.conv7(output)), 0.2)
        output = F.leaky_relu(self.bn8(self.conv8(output)), 0.2)

        output = self.global_pooling(output)
        output = F.leaky_relu(self.post_pooling_conv(output), 0.2)
        return F.sigmoid(self.output_conv(output).view(batch_size))