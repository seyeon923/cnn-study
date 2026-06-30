import lightning as L
import torch
from torch import nn
from torchmetrics.classification import Accuracy


class LitClassifier(L.LightningModule):
    def __init__(
        self, model: nn.Module, optimizer=torch.optim.AdamW, lr_scheduler=None
    ):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.criterion = nn.CrossEntropyLoss()

        self.num_classes = getattr(model, "output_classes")

        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

    def training_step(self, batch, batch_idx: int):
        del batch_idx

        image, target = batch

        pred: torch.Tensor = self.model(image)
        loss = self.criterion(pred, target)

        self.train_acc.update(pred, target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx: int):
        del batch_idx

        image, target = batch

        pred: torch.Tensor = self.model(image)
        loss = self.criterion(pred, target)

        self.val_acc.update(pred, target)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int):
        del batch_idx

        image, target = batch

        pred: torch.Tensor = self.model(image)

        self.test_acc.update(pred, target)
        self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters())
        lr_scheduler = self.lr_scheduler(optimizer) if self.lr_scheduler else None

        if lr_scheduler is None:
            return optimizer
        else:
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
