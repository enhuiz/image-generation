import json
import logging
import os
import random
from itertools import count

import numpy as np
import torch
import torchvision.transforms.functional as VF
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from tqdm import tqdm

from .config import cfg
from .fid import compute_fid
from .models import get_discriminator, get_generator
from .utils import Diagnostic, setup_logging, to_device, trainer

_logger = logging.getLogger(__name__)


def load_engines():
    generator = get_generator()

    engines = dict(
        generator=trainer.Engine(
            model=generator,
            config=cfg.gen_ds_cfg,
        ),
    )

    discriminator = get_discriminator()

    if discriminator is not None:
        engines["discriminator"] = trainer.Engine(
            model=discriminator,
            config=cfg.dis_ds_cfg,
        )

    return trainer.load_engines(engines, cfg)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    setup_logging(cfg.log_dir)

    if cfg.dataset.lower() == "mnist":
        Dataset = MNIST
    elif cfg.dataset.lower() == "cifar10":
        Dataset = CIFAR10
    else:
        raise NotImplementedError(cfg.dataset)

    train_ds = Dataset(
        str(cfg.data_dir),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    test_ds = Dataset(
        str(cfg.data_dir),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        worker_init_fn=_seed_worker,
    )

    train_200_dl = DataLoader(
        Subset(train_ds, [*range(200)]),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        drop_last=False,
        worker_init_fn=_seed_worker,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        drop_last=False,
        worker_init_fn=_seed_worker,
    )

    del train_ds, test_ds

    diagnostic = fake = None

    def train_feeder(engines, batch, name):
        nonlocal diagnostic, fake

        real, _ = batch

        generator = engines["generator"]
        discriminator = engines.get("discriminator", None)

        if name == "generator":
            if diagnostic is None:
                diagnostic = Diagnostic(generator.module)

            if trainer.get_cmd() == "diag-stop":
                diagnostic.save()

            if trainer.get_cmd() == "diag-start":
                diagnostic.attach()

            fake = generator(real)
            losses = generator.gather_attribute("loss")

            if discriminator is None:
                del fake
            else:
                discriminator.freeze()
                _ = discriminator(fake)
                losses |= discriminator.gather_attribute("loss")
                discriminator.unfreeze()
                fake = fake.detach()

        elif name == "discriminator":
            _ = discriminator(fake, real)
            losses = discriminator.gather_attribute("loss")
        else:
            raise NotImplementedError(name)

        loss = sum(losses.values())
        assert isinstance(loss, Tensor)

        stats = {}
        stats |= {k: v.item() for k, v in losses.items()}
        stats |= engines.gather_attribute("scalar")

        return loss, stats

    @torch.inference_mode()
    def run_eval(engines, name, dl):
        log_dir = cfg.log_dir / str(engines.global_step) / name

        generator = engines["generator"]

        counter = count()

        for batch in tqdm(dl):
            batch = to_device(batch, cfg.device)

            real, _ = batch

            if "vae" in cfg.generator:
                fake = generator(real, use_prior=True)
            else:
                fake = generator(real)

            for i, ri, fi in zip(counter, real, fake):
                real = VF.to_pil_image(ri.cpu())
                real_path = log_dir / "real" / f"{i:06d}.png"
                real_path.parent.mkdir(parents=True, exist_ok=True)
                real.save(real_path)

                fake = VF.to_pil_image(fi.cpu())
                fake_path = log_dir / "fake" / f"{i:06d}.png"
                fake_path.parent.mkdir(parents=True, exist_ok=True)
                fake.save(fake_path)

        fid = compute_fid([log_dir / "real", log_dir / "fake"])

        with open(log_dir / "fid.txt", "w") as f:
            f.write(str(fid))

        def _clean_dir(png_dir, num_kept_as_demo):
            for i, path in enumerate(sorted(png_dir.glob("*.png"))):
                if i >= num_kept_as_demo:
                    os.remove(path)

        _clean_dir(log_dir / "real", num_kept_as_demo=10)
        _clean_dir(log_dir / "fake", num_kept_as_demo=10)

        stats = {}
        stats["global_step"] = engines.global_step
        stats["name"] = name
        stats["fid"] = fid

        _logger.info(f"{json.dumps(stats)}.")

    def eval_fn(engines):
        run_eval(engines, "train_200", train_200_dl)
        run_eval(engines, "test", test_dl)

    trainer.train(
        engines_loader=load_engines,
        train_dl=train_dl,
        train_feeder=train_feeder,
        eval_fn=eval_fn,
    )


if __name__ == "__main__":
    main()
