from dataclasses import dataclass, field

from .component import ComponentConfig


@dataclass(frozen=True, slots=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    optimizer: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(name="AdamW")
    )
    lr_scheduler: ComponentConfig | None = None
