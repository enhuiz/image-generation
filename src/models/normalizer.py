"""
A data normalization layer, avoid pre-calculation of data statistics.

MIT License

Copyright (c) 2023 Zhe Niu

niuzhe.nz@outlook.com
"""

import logging
from typing import overload

import torch
from torch import Tensor, nn

_logger = logging.getLogger(__name__)


class Normalizer(nn.Module):
    def __init__(
        self,
        momentum=0.01,
        log_scale=False,
        eps=1e-9,
        num_channels=1,
        axis=None,
    ):
        super().__init__()
        self.momentum = momentum
        self.log_scale = log_scale
        self.eps = eps
        self.axis = axis
        self.running_mean: Tensor
        self.running_var: Tensor
        self.register_buffer("running_mean", torch.full([num_channels], float("nan")))
        self.register_buffer("running_var", torch.full([num_channels], float("nan")))

    @property
    def running_std(self):
        return (self.running_var + self.eps).sqrt()

    @torch.no_grad()
    def _ema(self, a: Tensor, x: Tensor):
        if a.isnan().any():
            return x
        return (1 - self.momentum) * a + self.momentum * x

    def _view_as(self, x, y):
        shape = tuple(n if i == self.axis else 1 for i, n in enumerate(y.shape))
        return x.view(shape)

    @torch.no_grad()
    def update_stats_(self, x, m):
        if not self.training:
            return

        if m is None:
            m = torch.ones_like(x)
        else:
            m = torch.ones_like(x) * m

        x = x.masked_fill(m == 0, float("nan"))

        if self.axis is None:
            x = x.flatten()
            x = x.unsqueeze(0)
        else:
            x = x.transpose(0, self.axis)
            x = x.flatten(1)

        nanmean = x.nanmean(dim=1, keepdim=True)
        nanvar = (x - nanmean).pow(2).nanmean(dim=1)

        self.running_mean = self._ema(self.running_mean, nanmean.squeeze(1))
        self.running_var = self._ema(self.running_var, nanvar)

        running_mean = self.running_mean.flatten().tolist()
        running_var = self.running_var.flatten().tolist()

        self.scalar = {}

        for i, (rmi, rvi) in enumerate(zip(running_mean, running_var)):
            self.scalar |= {f"{i}.mean": rmi, f"{i}.var": rvi}

    def forward(self, x: Tensor | None, m: Tensor | None = None, update_stats=True):
        """
        Args:
            x: values to be used for estimation of stats and normalized
            m: only m=1 value will be used to estimate stats
        Return:
            normalized x, masked
        """
        if x is None:
            return x

        if update_stats:
            self.update_stats_(x, m)

        if self.running_mean.isnan().any():
            _logger.warning(
                "Normalizer encounter NaN. Did you initialize? "
                f"Skip normalization anyway."
            )
            return x

        if self.log_scale:
            x = x.log1p()

        running_mean = self._view_as(self.running_mean, x)
        running_std = self._view_as(self.running_std, x)

        x = (x - running_mean) / running_std

        return x

    @overload
    def denormalize(self, x: Tensor) -> Tensor:
        ...

    @overload
    def denormalize(self, x: None) -> None:
        ...

    def denormalize(self, x: Tensor | None):
        if x is None:
            return x

        running_mean = self._view_as(self.running_mean, x)
        running_std = self._view_as(self.running_std, x)

        x = x * running_std + running_mean

        if self.log_scale:
            x = x.expm1()

        return x


if __name__ == "__main__":
    normalizer = Normalizer()
    x = torch.randn(3, 3, 1)
    m = (torch.randn(3, 3, 1) > 0).float()
    y = normalizer(x, m)
    print(x[m > 0].mean(), x[m > 0].var(unbiased=False))
    print(normalizer.scalar)

    normalizer = Normalizer(num_channels=3, axis=1)
    m = (torch.randn(3, 3, 1) > 0).float()
    print(x * m)
    y = normalizer(x, m)
    print(x[m > 0].mean(), x[m > 0].var(unbiased=False))
    print(normalizer.scalar)
