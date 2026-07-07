import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToTensor,
)

from ...utils import calculate_mean_std

CIFAR10_URL = (
    "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/cifar-10-python.tar.gz"
)


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        normalize: bool = True,
        augmentation: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.augmentation = augmentation

        self.mean = None
        self.std = None

    def prepare_data(self):
        CIFAR10.url = CIFAR10_URL
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

        if self.normalize:
            loader = DataLoader(
                CIFAR10(self.data_dir, train=True, transform=ToTensor()), self.batch_size
            )
            self.mean, self.std = calculate_mean_std(map(lambda x: x[0], loader))

    def setup(self, stage: str):
        train_transforms = [ToTensor()]
        val_transforms = [ToTensor()]

        if self.normalize:
            train_transforms.append(Normalize(self.mean, self.std))
            val_transforms.append(Normalize(self.mena, self.std))

        if self.augmentation:
            train_transforms.extend(
                [RandomHorizontalFlip(), RandomVerticalFlip(), RandomGrayscale()]
            )

        self.train_transform = Compose(train_transforms)
        self.val_transform = Compose(val_transforms)

        test_full = CIFAR10(self.data_dir, train=False, transform=self.val_transform)
        self.val_ds, self.test_ds = random_split(
            test_full, [0.5, 0.5], generator=torch.Generator().manual_seed(42)
        )

        self.classes = test_full.classes
        self.class_to_idx = test_full.class_to_idx

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_ds = CIFAR10(self.data_dir, train=True, transform=self.train_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
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


if __name__ == "__main__":
    data = CIFAR10DataModule("./data")
    data.prepare_data()
    data.setup("fit")

    print(data.classes)
    print(data.class_to_idx)
    print(data.mean, data.std)
