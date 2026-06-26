from dataclasses import dataclass, field
from typing import Any

from cnn_study.config.component import ComponentConfig


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str = "MNIST"
    root: str = "./data"

    num_classes: int = None

    extra_params: dict[str, Any] = field(default_factory=dict)

    transforms: list[ComponentConfig] = None
    target_transforms: list[ComponentConfig] = None
