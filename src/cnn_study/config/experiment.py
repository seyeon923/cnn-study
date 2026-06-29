from dataclasses import dataclass, field

from omegaconf import DictConfig, OmegaConf


@dataclass(slots=True)
class ExperimentConfig:
    lightning_module: DictConfig = field(default_factory=OmegaConf.create)
    seed: int = 42
