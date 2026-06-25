from pathlib import Path

import dacite
import yaml


def load_yaml_config(path: str | Path, data_class: type):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return dacite.from_dict(
        data_class=data_class, data=data, config=dacite.Config(strict=True)
    )
