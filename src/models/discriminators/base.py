from torch import nn

from ..normalizer import Normalizer


class DiscriminatorBase(nn.Module):
    def __init__(self):
        super().__init__()
        # Just use normalizer by default
        self.normalizer = Normalizer(num_channels=3, axis=1)

    def forward(self, fake, real=None):
        loss = {}
        fake = self.normalizer(fake, update_stats=False)
        if real is None:
            loss["g_fake"] = self._forward_impl(fake, 1)
        else:
            real = self.normalizer(real)
            loss["d_fake"] = self._forward_impl(fake, 0)
            loss["d_real"] = self._forward_impl(real, 1)
        self.loss = loss
        return loss

    def _forward_impl(self, x, s):
        raise NotImplementedError
