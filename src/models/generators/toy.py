import torch
from torch import nn


class ToyGenerator(nn.Module):
    def __init__(self, out_channels=3, num_channels=64):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels, out_channels, 1),
        )

    def forward(self, x):
        z = torch.randn(len(x), 1, 4, 4, device=x.device)
        del x
        x = self.seq(z)  # 32x32
        return x
