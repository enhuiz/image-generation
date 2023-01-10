import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.distributions import RelaxedOneHotCategorical, Categorical

from ....config import cfg
from ....utils.trainer import get_iteration


class DCVQVAE(nn.Module):
    def __init__(self, out_channels=3, base_channels=16):
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
                    nn.AvgPool2d(3, 2, padding=1),
                )
            )

        self.encoder.append(nn.Conv2d(c[-1], cfg.vae_num_quants, 1))

        self.codebook = nn.Embedding(cfg.vae_num_quants, cfg.vae_code_dim)

        rc = [cfg.vae_code_dim] + [*reversed(c)]

        self.decoder = nn.Sequential()

        for i in range(num_layers):
            self.decoder.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(rc[i], rc[i + 1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(rc[i + 1]),
                    nn.GELU(),
                )
            )

        self.decoder.append(
            nn.Sequential(
                nn.Conv2d(rc[-2], rc[-1], 1),
                nn.Tanh(),
            ),
        )

    @property
    def tau(self):
        """
        Anneal from 1 to 0.1 exponential
        """
        itr = get_iteration()

        max_tau = 1
        min_tau = 0.1

        max_log_tau = math.log(max_tau)
        min_log_tau = math.log(min_tau)

        if itr is None:
            return min_tau

        k = (min_log_tau - max_log_tau) / cfg.max_iter
        tau = math.exp(k * itr + max_log_tau)

        return tau

    def forward(self, x, use_prior=False):
        self.loss = {}
        self.scalar = dict(tau=self.tau)

        p_logits = torch.zeros(len(x), 2, 2, cfg.vae_num_quants, device=x.device)
        p = RelaxedOneHotCategorical(temperature=self.tau, logits=p_logits)

        if use_prior:
            weight = p.sample()
        else:
            logits = self.encoder(x)  # (b k h w)
            logits = rearrange(logits, "b k h w -> b h w k")
            q = RelaxedOneHotCategorical(temperature=self.tau, logits=logits)
            weight = q.rsample()

            self.loss["diversity"] = (
                Categorical(probs=logits.softmax(dim=-1).mean(dim=(0, 1, 2)))
                .entropy()
                .neg()
            )

        z = torch.einsum("b h w k, k d -> b d h w", weight, self.codebook.weight)

        h = self.decoder(z) * 3  # [-3, 3]

        self.loss["mse"] = self._reduce(F.mse_loss(h, x, reduction="none"))

        return h

    def _reduce(self, x, scale=True):
        # Sum all but average batch
        x = x.flatten(1).sum(1).mean()
        if scale:
            x = cfg.vae_reduction_scale * x
        return x
