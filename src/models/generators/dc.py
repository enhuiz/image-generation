import math

import torch
from torch import nn


class DCGenerator(nn.Module):
    def __init__(self, out_channels=3, base_channels=16):
        super().__init__()

        num_layers = int(math.log2(16))
        c = [out_channels] + [base_channels * 2**i for i in range(num_layers)]
        rc = [1] + [*reversed(c)]

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

        self.decoder.append(nn.Conv2d(rc[-2], rc[-1], 1))

    def forward(self, x):
        z = torch.randn(len(x), 1, 2, 2, device=x.device)
        del x
        x = self.decoder(z)  # 32x32
        return x
