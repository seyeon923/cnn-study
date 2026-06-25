from dataclasses import dataclass, field

from . import LRSchedulerConfig, OptimizerConfig


@dataclass(frozen=True, slots=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(name="AdamW")
    )
    lr_scheduler: LRSchedulerConfig | None = None
