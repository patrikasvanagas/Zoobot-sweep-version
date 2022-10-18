import logging
from operator import truediv
import pandas as pd
import os
from pytorch_lightning.plugins.training_type import DDPPlugin
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_galaxy_datasets.galaxy_datamodule import GalaxyDataModule
from zoobot.pytorch.training import losses
from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.estimators import (
    efficientnet_standard,
    resnet_torchvision_custom,
)  # also resnet_detectron2_custom, imported below
import sys
sys.path.insert(0, "home/patrikas_v/to_zip/zoobot")
from zoobot.shared import label_metadata, schemas
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_galaxy_datasets.prepared_datasets import (
    DecalsDR5Dataset,
    decals_dr5_setup,
    GZ2Dataset,
    gz2_setup,
)
from pytorch_galaxy_datasets.galaxy_dataset import GalaxyDataset
from zoobot.pytorch.training import train_with_pytorch_lightning
device = torch.device("cuda")
question_answer_pairs = label_metadata.decals_dr5_ortho_pairs
dependencies = label_metadata.decals_ortho_dependencies
schema = schemas.Schema(question_answer_pairs, dependencies)
train_catalog, train_label_cols = decals_dr5_setup(
    root="/home/patrikas_v/to_zip/decals", train=True, download=False
)
val_catalog, test_label_cols = decals_dr5_setup(
    root="/home/patrikas_v/to_zip/decals", train=False, download=False
)

train_catalog = train_catalog.sample(800) ################# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
val_catalog = val_catalog.sample(200)

wandb_logger = WandbLogger(
    project="zoobot-pytorch-catalog-example",
    name=os.path.basename("/home/patrikas_v/to_zip/zoobot/zoobot/pytorch/examples"),
    log_model="all",
)
train_with_pytorch_lightning.train_default_zoobot_from_scratch(
        save_dir = "/home/patrikas_v/to_zip/zoobot/zoobot/pytorch/examples",
        schema=schema,
        wandb_logger=wandb_logger,
        model_architecture='efficientnet',
        batch_size=64,
        epochs=1,
        patience=8,
        random_state=42,

        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=val_catalog,

        color=True,
        accelerator="gpu",
        nodes=1,
        gpus=2,
        num_workers=8,
        mixed_precision=False,
        add_default_albumentations=False,
        add_astroaugmentations = False,
        logger_pictures=True,
        resize_size=200,
        elastic_sigma = 100,
        elastic_alpha_affine = 2,
        p_elastic = 0.5,
        sersic_extent_min=5,
        sersic_extent_max=80,
        sersic_max_number = 5,
        p_sersic = 0.5,
        gaussian_extent_min=3,
        gaussian_extent_max=20,
        gaussian_max_number = 5,
        p_gaussian = 0.5,
        p_croppedtemplateoverlap = 0.5,
        brightness_gradient_minimum = 0.5,
        brightness_gradient_noise = 0.2,
        p_brightness_gradient = 0.5,
        album_shift_limit = 0.1,
        album_scale_limit = 0.1,
        album_rotate_limit = 180,
        p_album_shiftscalerotate = 0.5,
        p_flip = 0.5,
        channelwisedropout_max_fraction = 0.2,
        channelwisedropout_min_width = 10,
        channelwisedropout_min_height = 10,
        channelwisedropout_max_holes = 0.5,
        p_channelwisedropout = 0.5,
        )
