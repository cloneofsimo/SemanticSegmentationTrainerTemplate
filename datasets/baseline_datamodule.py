from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
import torchvision

import pandas as pd
from glob import glob
from random import randint
from tqdm import tqdm

import os
import hydra

from .baseline_dataset import BaselineSSDataset


class BaselineSSDataModule(pl.LightningDataModule):
    """
    Baseline Datamodule for pytorch lightning trainer.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        self.num_workers = cfg.train.num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2), (0.8))]
        )

    def setup(self, stage=None):
        self.train_dataset = BaselineSSDataset(self.cfg)
        self.val_dataset = BaselineSSDataset(self.cfg)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
