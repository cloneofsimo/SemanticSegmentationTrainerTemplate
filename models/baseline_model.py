from omegaconf.dictconfig import DictConfig
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch import optim
import torchvision.utils as vutils

from random import randint

from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation.fcn import FCNHead


class BaselineSSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = fcn_resnet50(pretrained=True, progress=True)
        self.core.classifier = FCNHead(2048, 2)
        self.core._requires_grad = True

    def forward(self, x):
        yh = self.core(x)

        return yh["out"]
