import torch

from . import LRSchedulerConfig, OptimizerConfig

OPTIMIZER_REGISTRY: dict[str, type[torch.optim.Optimizer]] = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}

LR_SCHEDULER_REGISTRY: dict[str, type[torch.optim.lr_scheduler.LRScheduler]] = {
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
}


def build_optimizer(cfg: OptimizerConfig):
    cls = OPTIMIZER_REGISTRY[cfg.name]
    return cls(**cfg.params)


def build_lr_scheduler(cfg: LRSchedulerConfig):
    cls = LR_SCHEDULER_REGISTRY[cfg.name]
    return cls(**cfg.params)
