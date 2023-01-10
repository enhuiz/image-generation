from ..config import cfg

from . import generators
from . import discriminators


def get_generator():
    if cfg.generator == "dc":
        return generators.dc.DCGenerator(cfg.num_channels)
    elif cfg.generator == "dc-vae":
        return generators.vae.dc.DCVAE(cfg.num_channels)
    elif cfg.generator == "dc-vq-vae":
        return generators.vae.dcvq.DCVQVAE(cfg.num_channels)
    elif cfg.generator == "dc-vae-pic":
        return generators.vae.dc.DCVAE(cfg.num_channels, use_pic=True)

    raise NotImplementedError(cfg.generator)


def get_discriminator():
    if cfg.discriminator is None:
        return None
    elif cfg.discriminator == "dc":
        return discriminators.dc.DCDiscriminator(cfg.num_channels)

    raise NotImplementedError(cfg.discriminator)
