from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    name: str = "lenet"
    params: dict[str, Any] = field(default_factory=dict)
