import os
import platform
import hydra
from numpy.core.numeric import cross
from omegaconf import DictConfig
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning import loggers as pl_loggers

from models.baseline_plmodel import BaselineSSModelPl
from datasets.baseline_datamodule import BaselineSSDataModule


@hydra.main(config_path="config.yaml")
def train_main(cfg: DictConfig):
    model = BaselineSSModelPl(cfg)
    data = BaselineSSDataModule(cfg)

    logger = pl_loggers.TensorBoardLogger(
        save_dir=cfg.train.log_dir, version=cfg.train.version
    )
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=cfg.train.checkpoint_dir,
        save_top_k=cfg.train.save_top_k,
    )

    trainer = Trainer(
        # accelerator=None if platform.system() == "Windows" else "ddp",
        # accumulate_grad_batches=cfg.train.accumulate,
        # auto_scale_batch_size="binsearch",
        max_epochs=cfg.train.epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=cfg.train.log_dir,
        fast_dev_run=cfg.train.fast_dev_run,
        gpus=cfg.train.gpus,
        logger=logger,
        terminate_on_nan=True,
        weights_save_path=cfg.train.checkpoint_dir,
        # check_val_every_n_epoch=cfg.train.check_val_freq,
        val_check_interval=0.25,
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    train_main()
