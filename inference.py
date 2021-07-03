import os
import platform
import hydra
from numpy.core.numeric import cross
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning import loggers as pl_loggers

from models.baseline_plmodel import BaselineSSModelPl
from datasets.baseline_datamodule import BaselineSSDataModule
from datasets.baseline_dataset import BaselineSSDataset

import streamlit as st
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks

import torch


@st.cache(allow_output_mutation=True)
def _load_all(cfg):
    model = BaselineSSModelPl(cfg)

    checkpoint_path = "/home/taeho/Competition/siim-covid19-detection/simo_workspace_od/Pytorch_Segmentation_Template/outputs/2021-07-03/21-56-28/checkpoints/epoch=3-step=42.ckpt"

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


if __name__ == "__main__":
    cfg = OmegaConf.load("./config.yaml")
    device = "cuda:1"
    model = _load_all(cfg)
    model.to(device)
    dset = BaselineSSDataset(cfg)
    idx = st.slider("IDX", min_value=0, max_value=len(dset), value=0)

    data = dset[idx]
    seg = model.core(data["x"].unsqueeze(dim=0).to(device))
    img = data["x"]
    print((seg[0, 1:2, :, :] > 0.2).type(torch.bool).detach().cpu())
    segimg = draw_segmentation_masks(
        (img * 255).type(torch.uint8),
        (seg[0, 1:2, :, :] > 0.2).detach().cpu(),
    )

    segimg = F.to_pil_image(segimg)
    st.image(segimg)
