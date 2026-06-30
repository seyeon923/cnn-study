from dataclasses import dataclass, field

from omegaconf import DictConfig, OmegaConf


@dataclass(slots=True)
class ExperimentConfig:
    lightning_module: DictConfig = field(default_factory=OmegaConf.create)
    data_module: DictConfig = field(
        default_factory=lambda: OmegaConf.create(
            {
                "_target_": "cnn_study.data.data_modules.mnist_data_module.MNISTDataModule",
                "data_dir": "./data",
            }
        )
    )
