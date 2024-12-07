import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_chanels, out_chanels):
        super().__init__()
        self.conv = DoubleConv(in_chanels, out_chanels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)

        return skip, down


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_1 = DownSample(1, 32)
        self.down_2 = DownSample(32, 64)
        self.down_3 = DownSample(64, 128)
        # self.down_4 = DownSample(in_chanels, out_chanels)

        self.bottom = DoubleConv(128, 256)

        self.up_1 = UpSample(256, 128)
        self.up_2 = UpSample(128, 64)
        self.up_3 = UpSample(64, 32)
        # self.up_4 = UpSample(in_chanels, out_chanels)

        self.out = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        s1, d1 = self.down_1(x)
        s2, d2 = self.down_2(d1)
        s3, d3 = self.down_3(d2)

        b = self.bottom(d3)

        u1 = self.up_1(b, s3)
        u2 = self.up_2(u1, s2)
        u3 = self.up_3(u2, s1)

        y = self.out(u3)
        return y
