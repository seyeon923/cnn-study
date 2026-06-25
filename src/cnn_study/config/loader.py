from pathlib import Path

import dacite
import yaml

from .component import ComponentConfig


def load_config(path: str | Path, data_class: type):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return dacite.from_dict(
        data_class=data_class, data=data, config=dacite.Config(strict=True)
    )


def load_component_config(path: str | Path):
    return load_config(path, ComponentConfig)


if __name__ == "__main__":
    lenet5_cfg = load_component_config("./configs/models/lenet5.yaml")
    print(lenet5_cfg)

    alexnet_cfg = load_component_config("./configs/models/alexnet.yaml")
    print(alexnet_cfg)
