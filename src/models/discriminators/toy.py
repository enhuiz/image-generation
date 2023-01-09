from torch import nn
from torch.nn.utils.spectral_norm import spectral_norm

from .base import DiscriminatorBase


class ToyDilatedModel(DiscriminatorBase):
    def __init__(self, in_channels=3, num_channels=64):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels, num_channels, 3, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels, num_channels, 3, dilation=4, padding=4, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels, num_channels, 3, dilation=8, padding=8, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels, 1, 1),
        )

        self.apply(
            lambda m: (lambda _: None)(spectral_norm(m))
            if isinstance(m, nn.Conv2d)
            else None
        )

    def _forward_impl(self, x, s):
        h = self.seq(x)  # (b 1 h w)
        return (h.flatten(1) - s).square().mean()
