import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, kl_divergence

from ....config import cfg
from .pic import PIC


class ToyVAE(nn.Module):
    def __init__(self, out_channels=3, num_channels=64, use_pic=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(out_channels, num_channels, 3, padding=1, bias=False),
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
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Conv2d(num_channels, 2, 1),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
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

        self.use_pic = use_pic

    def forward(self, x, use_prior=False):
        self.loss = {}
        self.scalar = {}

        if use_prior:
            z = torch.randn(len(x), 1, 2, 2, device=x.device)
        else:
            μ, logσ = self.encoder(x).chunk(2, dim=1)
            σ = logσ.exp()
            q = Normal(μ, σ)
            p = Normal(torch.zeros_like(μ), torch.ones_like(σ))
            kl = self._reduce(kl_divergence(q, p))
            if self.use_pic:
                beta = self.pic(kl)
            else:
                beta = cfg.vae_default_beta
            self.loss["kl"] = beta * kl
            self.scalar["kl"] = kl.item()
            z = q.rsample()

        h = self.decoder(z)

        self.loss["mse"] = self._reduce(F.mse_loss(h, x, reduction="none"))

        return h

    def _reduce(self, x):
        # Sum all channels, average over batch and locations
        return x.sum(dim=1).mean()
