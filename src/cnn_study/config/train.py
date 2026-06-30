from dataclasses import dataclass, field

from omegaconf import DictConfig, OmegaConf


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 100
    optimizer: DictConfig = field(
        default_factory=lambda: OmegaConf.create({"_target_": "torch.optim.AdamW"})
    )
    lr_scheduler: DictConfig = field(default_factory=OmegaConf.create)
