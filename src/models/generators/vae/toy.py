import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, kl_divergence

from .pic import PIC


class ToyDilatedVAE(nn.Module):
    def __init__(self, out_channels=3, num_channels=64, use_pic=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(out_channels, num_channels, 3, padding=1, bias=False),
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
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(num_channels, 2, 1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(1, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels, out_channels, 1),
        )

        if use_pic:
            self.pic = PIC()

    def forward(self, x, use_prior=False):
        if use_prior:
            z = torch.randn(len(x), 1, 4, 4, device=x.device)
        else:
            μ, logσ = self.encoder(x).chunk(dim=1)
            σ = logσ.exp()
            q = Normal(μ, σ)
            p = Normal(torch.zeros_like(μ), torch.ones_like(σ))
            kl = self._reduce(kl_divergence(q, p))
            beta = self.pic(kl)
            self.loss = dict(kl=beta * kl)
            z = q.rsample()

        h = self.decoder(z)

        self.loss["mse"] = self._reduce(F.mse_loss(h, x, reduction="none"))

        return h

    def _reduce(self, x):
        # Sum all channels, average over batch and locations
        return x.sum(dim=1).mean()
