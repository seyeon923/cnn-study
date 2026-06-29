import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from cnn_study.config import ExperimentConfig, TrainConfig

cs = ConfigStore.instance()
cs.store(name="base_experiment", node=ExperimentConfig)
cs.store(group="lightning_module/train_cfg", name="base_train_cfg", node=TrainConfig)


@hydra.main(version_base=None, config_path="../configs", config_name="experiment")
def train(cfg: ExperimentConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    train()
