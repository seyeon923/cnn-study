from pathlib import Path

import dacite
import yaml


def load_config(path: str | Path, data_class: type):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return dacite.from_dict(
        data_class=data_class, data=data, config=dacite.Config(strict=True)
    )


if __name__ == "__main__":
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    lenet5_cfg = OmegaConf.load("./configs/models/lenet5.yaml")
    print(lenet5_cfg)
    lenet5 = instantiate(lenet5_cfg)
    print(lenet5)

    alexnet_cfg = OmegaConf.load("./configs/models/alexnet.yaml")
    print(alexnet_cfg)
    alexnet = instantiate(alexnet_cfg)
    print(alexnet)
