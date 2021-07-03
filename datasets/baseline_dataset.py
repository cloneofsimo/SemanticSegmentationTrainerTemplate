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

from PIL import Image
import torchvision.transforms as T
from omegaconf import OmegaConf

import numpy as np


class BaselineSSDataset(Dataset):
    def __init__(
        self, cfg: DictConfig, transform=T.Compose([T.ToTensor(), T.Resize((256, 256))], seed = 0, ratio = 0)
    ):
        self.data_root = cfg.data.root
        self.imgs = sorted(glob(cfg.data.root + cfg.data.img_folder))
        self.masks = sorted(glob(cfg.data.root + cfg.data.mask_folder))
        self.transform = transform


    def _get_idx_file(self, idx):
        """
        Custom filepath algorithm. For PennFudanDataset, this is...
        """
        img_path, mask_path = self.imgs[idx], self.masks[idx]

        return img_path, mask_path

    def __getitem__(self, idx):

        img_path, mask_path = self._get_idx_file(idx)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = T.Resize((256, 256))(torch.tensor(np.array(mask)).unsqueeze(dim=0))

        # In case of PennFudan Dataset, you'll want to remove different classes (as they are also instance segmentation dataset.)
        mask = (mask > 0).int()

        
        img = self.transform(img)

        data = {"x": img, "y": mask}
        return data

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":

    # visualize with streamlit. It is good.

    import streamlit as st
    import torchvision.transforms.functional as F
    from torchvision.utils import draw_segmentation_masks

    cfg = OmegaConf.load(
        "../config.yaml"
    )

    dset = BaselineSSDataset(cfg)
    idx = st.slider("IDX", min_value=0, max_value=len(dset), value=0)
    data = dset[idx]
    img, mask = data["x"], data["y"]

    print((mask).type(torch.bool).shape)

    segimg = draw_segmentation_masks(
        (img * 255).type(torch.uint8), (mask).type(torch.bool)
    )

    segimg = F.to_pil_image(segimg)
    st.image(segimg)
