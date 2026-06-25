from dataclasses import dataclass

from . import ModelConfig
from .train import TrainConfig


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model: ModelConfig
    train: TrainConfig
    seed: int = 42
