import hydra
import lightning as L
import numpy as np
import ordinal
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from cnn_study.config import Config


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: Config):
    model: L.LightningModule = instantiate(cfg.lightning_module)
    datamodule: L.LightningDataModule = instantiate(cfg.data)

    best_scores: dict[str, list[float]] = {}
    for i in range(cfg.train_repeats):
        print(f"Start {ordinal.ordinal(i + 1)} training")
        trainer: L.Trainer = instantiate(cfg.trainer)

        trainer.fit(model, datamodule=datamodule)

        cur_best_scores: dict[str, int] = {}

        for ckpt in trainer.checkpoint_callbacks:
            monitor = ckpt.monitor
            best_score = ckpt.best_model_score

            print(f"Best {monitor}: {best_score}")
            print(f"Best {monitor}'s checkpoint was saved to '{ckpt.best_model_path}'")

            if monitor not in cur_best_scores:
                cur_best_scores[monitor] = best_score

        for m, s in cur_best_scores.items():
            s = s.item()
            if i == 0:
                best_scores[m] = [s]
            else:
                best_scores[m].append(s)

    if cfg.train_repeats > 1:
        output_dir = HydraConfig.get().runtime.output_dir

        metrics = []
        means = []
        stds = []
        mins = []
        maxes = []

        for metric, scores in best_scores.items():
            pd.DataFrame(scores, columns=[metric]).to_csv(f"{output_dir}/{metric}.csv")

            metrics.append(metric)
            means.append(np.mean(scores))
            stds.append(np.std(scores, ddof=1))
            mins.append(np.min(scores))
            maxes.append(np.max(scores))

        pd.DataFrame(
            {
                "Metric": metrics,
                "Mean": means,
                "Std": stds,
                "Min": mins,
                "Max": maxes,
                "Num Repeats": [cfg.train_repeats] * len(metrics),
            }
        ).to_csv(f"{output_dir}/best_metrics_stats.csv", index=False)


if __name__ == "__main__":
    train()
