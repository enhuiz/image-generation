"""
A data normalization layer, avoid pre-calculation of data statistics.

MIT License

Copyright (c) 2023 Zhe Niu

niuzhe.nz@outlook.com
"""

import torch
from torch import nn, Tensor

from ....config import cfg


class PIC(nn.Module):
    """
    https://arxiv.org/pdf/2004.05988.pdf
    """

    def __init__(
        self,
        target: float = cfg.vae_target_kl,
        k_p=1e-2,
        k_i=1e-3,
        momentum=0.3,
    ):
        super().__init__()
        self.target = target
        self.k_i = k_i
        self.k_p = k_p
        self.error: Tensor
        self.integral: Tensor
        self.momentum = momentum
        self.register_buffer("error", torch.zeros([]))
        self.register_buffer("integral", torch.zeros([]))

    def _ema(self, e):
        self.error = (1 - self.momentum) * self.error + self.momentum * e
        return self.error

    @torch.no_grad()
    def forward(self, value: Tensor):
        e = value - self.target
        e = self._ema(e)

        # If e is > 0, increase y
        # if e is < 0, decrease y

        # Calculate the proportional term
        p = self.k_p / (1.0 + e.neg().exp())

        # Calculate the integral term
        i = self.integral + self.k_i * e

        # Calculate the control output
        y = p + i

        # Clamp the result
        y = y.clamp(cfg.vae_min_beta, cfg.vae_max_beta)

        # Update the integral
        self.integral.fill_(y - p)

        self.scalar = dict(p=p.item(), i=i.item(), y=y.item())

        return y
