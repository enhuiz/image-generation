from ..config import cfg

from . import generators
from . import discriminators


def get_generator():
    if cfg.generator == "toy":
        return generators.toy.ToyGenerator(cfg.num_channels)
    elif cfg.generator == "toy-vae":
        return generators.vae.toy.ToyVAE(cfg.num_channels)
    elif cfg.generator == "toy-vae-pic":
        return generators.vae.toy.ToyVAE(cfg.num_channels, use_pic=True)

    raise NotImplementedError(cfg.generator)


def get_discriminator():
    if cfg.discriminator is None:
        return None
    elif cfg.discriminator == "toy":
        return discriminators.toy.ToyModel(cfg.num_channels)

    raise NotImplementedError(cfg.discriminator)
