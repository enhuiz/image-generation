from torch import nn
from torch.nn.utils.weight_norm import weight_norm

from .base import DiscriminatorBase


class ToyModel(DiscriminatorBase):
    def __init__(self, in_channels=3, num_channels=64):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels, 1, 1),
        )

        self.apply(
            lambda m: (lambda _: None)(weight_norm(m))
            if isinstance(m, nn.Conv2d)
            else None
        )

    def _forward_impl(self, x, s):
        h = self.seq(x)  # (b 1 h w)
        return (h.flatten(1) - s).square().mean()
