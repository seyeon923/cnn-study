import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from ...utils import calculate_mean_std


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        normalize: bool = True,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.mean = None
        self.std = None

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

        if self.normalize:
            loader = DataLoader(
                MNIST(self.data_dir, train=True, transform=ToTensor()), self.batch_size
            )
            self.mean, self.std = calculate_mean_std(x[0] for x in loader)

    def setup(self, stage: str):
        if self.normalize:
            self.transform = Compose([ToTensor(), Normalize(self.mean, self.std)])
        else:
            self.transform = ToTensor()

        test_full = MNIST(self.data_dir, train=False, transform=self.transform)
        self.val_ds, self.test_ds = random_split(
            test_full, [0.5, 0.5], generator=torch.Generator().manual_seed(42)
        )

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_ds = MNIST(self.data_dir, train=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
        )
