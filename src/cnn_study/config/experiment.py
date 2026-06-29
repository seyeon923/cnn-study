from dataclasses import dataclass, field

from omegaconf import DictConfig, OmegaConf

from .train import TrainConfig


@dataclass(slots=True)
class ExperimentConfig:
    model: DictConfig = field(
        default_factory=lambda: OmegaConf.create(
            {"_target_": "cnn_study.models.lenet.LeNet5"}
        )
    )
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42
