import math

from torch import nn
from torch.nn.utils.spectral_norm import spectral_norm

from .base import DiscriminatorBase


class DCDiscriminator(DiscriminatorBase):
    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()

        num_layers = int(math.log2(16))
        c = [in_channels] + [base_channels * 2**i for i in range(num_layers)]

        self.encoder = nn.Sequential()

        for i in range(num_layers):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(c[i], c[i + 1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(c[i + 1]),
                    nn.GELU(),
                    nn.AvgPool2d(3, 2),
                )
            )

        self.encoder.append(nn.Conv2d(c[-1], 1, 1))

        self.apply(
            lambda m: (lambda _: None)(spectral_norm(m))
            if isinstance(m, nn.Conv2d)
            else None
        )

    def _forward_impl(self, x, s):
        h = self.encoder(x)  # (b 1 h w)
        return (h.flatten(1) - s).square().mean()
