import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, kl_divergence

from ....config import cfg
from .pic import PIC


class DCVAE(nn.Module):
    def __init__(self, out_channels=3, base_channels=16, use_pic=False):
        super().__init__()

        num_layers = int(math.log2(16))

        c = [out_channels] + [base_channels * 2**i for i in range(num_layers)]

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

        self.encoder.append(nn.Conv2d(c[-1], 2, 1))

        rc = [1] + [*reversed(c)]

        self.decoder = nn.Sequential()

        for i in range(num_layers):
            self.decoder.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(rc[i], rc[i + 1], 3, bias=False),
                    nn.BatchNorm2d(rc[i + 1]),
                    nn.GELU(),
                )
            )

        self.decoder.append(nn.Conv2d(rc[-2], rc[-1], 1))

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
            kl = self._reduce(kl_divergence(q, p), scale=False)
            if self.use_pic:
                beta = self.pic(kl).item()
            else:
                beta = cfg.vae_default_beta
            self.loss["kl"] = cfg.vae_reduction_scale * beta * kl
            self.scalar["kl"] = kl.item()
            self.scalar["beta"] = beta
            z = q.rsample()

        h = self.decoder(z)

        self.loss["mse"] = self._reduce(F.mse_loss(h, x, reduction="none"))

        return h

    def _reduce(self, x, scale=True):
        # Sum all but average batch
        x = x.flatten(1).sum(1).mean()
        if scale:
            x = cfg.vae_reduction_scale * x
        return x
