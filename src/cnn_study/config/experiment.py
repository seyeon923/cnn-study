from dataclasses import dataclass

from omegaconf import DictConfig

from .train import TrainConfig


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model: DictConfig
    train: TrainConfig
    seed: int = 42
