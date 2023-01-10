from torch import nn


class DiscriminatorBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake, real=None):
        loss = {}
        if real is None:
            loss["g_fake"] = self._forward_impl(fake, 1)
        else:
            loss["d_fake"] = self._forward_impl(fake, 0)
            loss["d_real"] = self._forward_impl(real, 1)
        self.loss = loss
        return loss

    def _forward_impl(self, x, s):
        raise NotImplementedError
