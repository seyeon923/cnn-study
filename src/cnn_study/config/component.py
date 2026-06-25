from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ComponentConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


if __name__ == "__main__":
    from .loader import load_yaml_config

    lenet5_cfg = load_yaml_config("./configs/models/lenet5.yaml", ComponentConfig)
    print(lenet5_cfg)

    alexnet_cfg = load_yaml_config("./configs/models/alexnet.yaml", ComponentConfig)
    print(alexnet_cfg)
