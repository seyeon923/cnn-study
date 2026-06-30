import hydra
import lightning as L
from hydra.utils import instantiate

from cnn_study.config import Config


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: Config):
    model: L.LightningModule = instantiate(cfg.lightning_module)
    datamodule: L.LightningDataModule = instantiate(cfg.data)
    trainer: L.Trainer = instantiate(cfg.trainer)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
