import torch

from ..models import build_model as __build_model
from .component import ComponentConfig

OPTIMIZER_REGISTRY: dict[str, type[torch.optim.Optimizer]] = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}

LR_SCHEDULER_REGISTRY: dict[str, type[torch.optim.lr_scheduler.LRScheduler]] = {
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
}


def build_model(cfg: ComponentConfig):
    return __build_model(cfg.name, **cfg.params)


def build_optimizer(cfg: ComponentConfig):
    cls = OPTIMIZER_REGISTRY[cfg.name]
    return cls(**cfg.params)


def build_lr_scheduler(cfg: ComponentConfig):
    if cfg is None:
        return None

    cls = LR_SCHEDULER_REGISTRY[cfg.name]
    return cls(**cfg.params)
