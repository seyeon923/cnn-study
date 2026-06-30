from dataclasses import dataclass, field

from omegaconf import DictConfig, OmegaConf


@dataclass(slots=True)
class Config:
    model: DictConfig = field(default_factory=OmegaConf.create)
    lightning_module: DictConfig = field(default_factory=OmegaConf.create)
    data: DictConfig = field(default_factory=OmegaConf.create)
    trainer: DictConfig = field(default_factory=OmegaConf.create)

    output_dir: str = "outputs"
    experiment_name: str = "exp"
