__all__ = [
    "ModelConfig",
    "OptimizerConfig",
    "LRSchedulerConfig",
    "TrainConfig",
    "ExperimentConfig",
]

from .component import ComponentConfig
from .experiment import ExperimentConfig
from .train import TrainConfig

ModelConfig = ComponentConfig
OptimizerConfig = ComponentConfig
LRSchedulerConfig = ComponentConfig
