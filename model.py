import math

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.prelu(y)
        y = self.conv2(y)

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

        residual_blocks_list = [ResidualBlock(base_filters) for _ in range(n_residual_blocks)]
        self.residual_blocks = nn.Sequential(*residual_blocks_list)

        self.post_residual_blocks = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters)
        )

        upsample_blocks_list = [UpsampleBlock(base_filters, 2) for _ in range(self.upsample_block_num)]
        self.upsample_block = nn.Sequential(*upsample_blocks_list)
        self.upsample_bicubic = nn.Upsample(scale_factor=upsample_factor, mode="bicubic", align_corners=True)

        self.last_conv = nn.Conv2d(base_filters, 3, kernel_size=9, padding=4)

    def forward(self, x):
        first_output = self.first_block(x)

        output = self.residual_blocks(first_output)

        output = self.post_residual_blocks(output)
        output = torch.add(output, first_output)

        output = self.upsample_block(output)

        return self.last_conv(output) + self.upsample_bicubic(x)


class Discriminator(nn.Module):
    def __init__(self, base_filters=64, patch_size=96):
        super().__init__()

        self.patch_size = patch_size
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
        self.bn7 = nn.BatchNorm2d(base_filters * 8)

        self.conv8 = nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(base_filters * 8)

        self.classifier = nn.Sequential(
            nn.Linear(base_filters * 8 * ((self.patch_size // 16) ** 2), 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

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
        output = torch.flatten(output, 1)
        return self.classifier(output).view(batch_size)

