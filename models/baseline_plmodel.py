from omegaconf.dictconfig import DictConfig
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch import optim
import torchvision.utils as vutils
from .baseline_model import *
from .criterions import *


class BaselineSSModelPl(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.core = BaselineSSModel()
        self.criterion = FocalLoss()

    def forward(self, x, y):
        yh = self.core(x)
        loss = self.criterion(yh, y)
        return yh, loss

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]

        _, loss = self.forward(x, y)

        return {"loss": loss}

    def training_epoch_end(self, outputs):

        loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log("train_loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):

        x, y = batch["x"], batch["y"]

        yh, loss = self.forward(x, y)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        if self.cfg.train.optim == "adam":
            return optim.Adam(
                self.core.parameters(),
                lr=self.cfg.train.lr,
                betas=self.cfg.train.betas,
                weight_decay=self.cfg.train.weight_decay,
                amsgrad=True,
            )
        else:
            raise NotImplementedError
