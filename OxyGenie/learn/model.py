import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image


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


class EUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down_1 = DownSample(1, 32)
        self.down_2 = DownSample(32, 64)
        self.down_3 = DownSample(64, 128)
        # self.down_4 = DownSample(in_chanels, out_chanels)

        self.embedding_layer = nn.Sequential(
            nn.Linear(2, 128),   # Embedding of size 2 to match 256 channels
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.bottom = DoubleConv(128, 128)

        self.up_1 = UpSample(256, 128)
        self.up_2 = UpSample(128, 64)
        self.up_3 = UpSample(64, 32)
        # self.up_4 = UpSample(in_chanels, out_chanels)

        self.out = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        s1, d1 = self.down_1(x1)
        s2, d2 = self.down_2(d1)
        s3, d3 = self.down_3(d2)

        b = self.bottom(d3)
        embedding = self.embedding_layer(x2)
        # Reshape to [batch_size, 256, 1, 1]
        embedding = embedding.unsqueeze(-1).unsqueeze(-1)
        # Match spatial dimensions
        embedding = embedding.expand(-1, -1, b.size(2), b.size(3))
        b_combined = torch.cat([b, embedding], 1)

        u1 = self.up_1(b_combined, s3)
        u2 = self.up_2(u1, s2)
        u3 = self.up_3(u2, s1)

        y = self.out(u3)
        return y

    @torch.no_grad
    def predict(self, img, params):
        test_image = img.copy()
        x_min, x_max = np.min(test_image), np.max(test_image)
        test_image = (test_image - x_min) / (x_max - x_min)
        test_image = Image.fromarray(test_image).resize((512, 512))

        im = TF.to_tensor(test_image).unsqueeze(0)

        params = params.copy()
        params[0] = params[0] / 10
        params[1] = params[1] * 100
        params = torch.tensor(params).unsqueeze(0)

        result_np = self(im, params).squeeze().numpy()
        return (result_np - np.min(result_np)) * 100
