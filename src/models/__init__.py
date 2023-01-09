from ..config import cfg

from . import generators
from . import discriminators


def get_generator():
    if cfg.generator == "toy":
        return generators.toy.ToyGenerator(cfg.num_channels)
    elif cfg.generator == "toy-dilated-vae":
        return generators.vae.toy.ToyDilatedVAE(cfg.num_channels)
    elif cfg.generator == "toy-dilated-vae-pic":
        return generators.vae.toy.ToyDilatedVAE(cfg.num_channels, use_pic=True)

    raise NotImplementedError(cfg.generator)


def get_discriminator():
    if cfg.discriminator is None:
        return None
    elif cfg.discriminator == "toy-dilated":
        return discriminators.toy.ToyDilatedModel(cfg.num_channels)

    raise NotImplementedError(cfg.discriminator)
