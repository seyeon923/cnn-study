from dataclasses import dataclass

from .component import ComponentConfig
from .train import TrainConfig


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model: ComponentConfig
    train: TrainConfig
    seed: int = 42
