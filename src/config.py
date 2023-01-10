import math
from dataclasses import dataclass
from pathlib import Path

from .utils.config import Config as ConfigBase


@dataclass(frozen=True)
class Config(ConfigBase):
    data_dir: Path = Path("data")
    dataset: str = "cifar10"

    @property
    def num_channels(self):
        if self.dataset.lower() in ["mnist"]:
            return 1
        return 3

    generator: str = "toy-dilated"
    discriminator: str | None = "toy-dilated"

    batch_size: int = 128
    eval_batch_size: int = 512
    warmup_min_lr: float = 1e-6
    warmup_max_lr: float = 2e-4
    dis_warmup_max_lr: float = 4e-4
    warmup_num_steps: int = 1_000
    max_iter: int = 10_000
    gradient_clipping: float = 100
    eval_every: int = 2_000
    save_ckpt_every: int = 10_000

    constant_lr: float | None = None

    vae_min_beta: float = 1e-9
    vae_max_beta: float = 1e3
    vae_target_kl: float = 3
    vae_default_beta: float = 1
    vae_reduction_scale: float = 1e-3
    vae_code_dim: int = 32
    vae_use_perceptual: bool = False

    @property
    def vae_num_quants(self):
        return round(math.exp(self.vae_target_kl))

    @property
    def gen_warmup_decay_scheduler(self):
        return {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": self.warmup_min_lr,
                "warmup_max_lr": self.warmup_max_lr,
                "warmup_num_steps": self.warmup_num_steps,
                "total_num_steps": self.max_iter,
                "warmup_type": "linear",
            },
        }

    @property
    def dis_warmup_decay_scheduler(self):
        return {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": self.warmup_min_lr,
                "warmup_max_lr": self.dis_warmup_max_lr,
                "warmup_num_steps": self.warmup_num_steps,
                "total_num_steps": self.max_iter,
                "warmup_type": "linear",
            },
        }

    @property
    def gen_ds_cfg(self):
        return {
            "train_micro_batch_size_per_gpu": self.batch_size,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "Adam",
                "betas": (0.0, 0.999),
            },
            "scheduler": self.gen_warmup_decay_scheduler,
            "gradient_clipping": self.gradient_clipping,
        }

    @property
    def dis_ds_cfg(self):
        return {
            "train_micro_batch_size_per_gpu": self.batch_size,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "Adam",
                "betas": (0.0, 0.999),
            },
            "scheduler": self.dis_warmup_decay_scheduler,
            "gradient_clipping": self.gradient_clipping,
        }


cfg = Config.from_cli()

if __name__ == "__main__":
    print(cfg)
