import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg19


class Vgg19(nn.ModuleList):
    def __init__(self):
        super().__init__()
        vgg_layers = [*vgg19(pretrained=True).features]

        self.slices = nn.ModuleList(
            [
                nn.Sequential(*vgg_layers[s:e])
                for s, e in [
                    [0, 2],
                    [2, 7],
                    [7, 12],
                    [12, 21],
                    [21, 30],
                ]
            ]
        )

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        out = []
        for slice in self.slices:
            x = slice(x)
            out.append(x.flatten(1))
        return torch.cat(out, dim=1)


class PerceptualFeatures(nn.Module):
    def __init__(self, scales=[1, 0.5]):
        super().__init__()
        self.scales = scales
        self.vgg19 = Vgg19()

    def pyramide(self, x):
        for scale in self.scales:
            yield F.interpolate(x, scale_factor=scale)

    def forward(self, x):
        out = []
        for scaled_x in self.pyramide(x):
            out.append(self.vgg19(scaled_x).flatten(1))
        return torch.cat(out, dim=1)


class PerceptualLoss:
    """
    Note that this is not a nn.Module.
    """

    def __init__(self):
        self.perceptual = PerceptualFeatures()

    def __call__(self, x, y):
        if next(self.perceptual.parameters()).device != x.device:
            self.perceptual.to(x.device)

        x = self.perceptual(x)
        y = self.perceptual(y)

        return F.l1_loss(x, y)
