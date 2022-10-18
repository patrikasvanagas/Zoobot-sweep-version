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
import wandb
import yaml
device = torch.device("cuda")
question_answer_pairs = label_metadata.decals_dr5_ortho_pairs
dependencies = label_metadata.decals_ortho_dependencies
schema = schemas.Schema(question_answer_pairs, dependencies)
train_catalog, _ = decals_dr5_setup(
    root="/home/patrikas_v/to_zip/decals", train=True, download=True
)
val_catalog, _ = decals_dr5_setup(
    root="/home/patrikas_v/to_zip/decals", train=False, download=True
)
train_catalog = train_catalog.sample(800) ################# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
val_catalog = val_catalog.sample(200)
with open("/home/patrikas_v/to_zip/zoobot/zoobot/pytorch/examples/config.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
run = wandb.init(config=config)
wandb_logger = WandbLogger(
    project="sweep 1",
    name=os.path.basename("/home/patrikas_v/to_zip/zoobot/zoobot/pytorch/examples"),
    log_model="all",
)
train_with_pytorch_lightning.train_default_zoobot_from_scratch(
    save_dir="/home/patrikas_v/to_zip/zoobot/zoobot/pytorch/examples",
    schema=schema,
    wandb_logger=wandb_logger,
    color=True,
    model_architecture="efficientnet",
    accelerator="gpu",
    nodes=1,
    num_workers=8,
    add_default_albumentations=wandb.config.add_default_albumentations,
    add_astroaugmentations=wandb.config.add_astroaugmentations,
    logger_pictures=wandb.config.logger_pictures,
    train_catalog=train_catalog,
    val_catalog=val_catalog,
    test_catalog=val_catalog,
    batch_size=wandb.config.batch_size,
    epochs=wandb.config.epochs,
    patience=wandb.config.patience,
    gpus=wandb.config.gpus,
    random_state=wandb.config.random_state,
    resize_size=wandb.config.resize_size,
    elastic_sigma=wandb.config.elastic_sigma,
    elastic_alpha_affine=wandb.config.elastic_alpha_affine,
    p_elastic=wandb.config.p_elastic,
    gaussian_extent_min=wandb.config.gaussian_extent_min,
    gaussian_extent_max=wandb.config.gaussian_extent_max,
    gaussian_max_number=wandb.config.gaussian_max_number,
    p_gaussian=wandb.config.p_gaussian,
    p_croppedtemplateoverlap=wandb.config.p_croppedtemplateoverlap,
    brightness_gradient_minimum=wandb.config.brightness_gradient_minimum,
    brightness_gradient_noise=wandb.config.brightness_gradient_noise,
    p_brightness_gradient=wandb.config.p_brightness_gradient,
    album_shift_limit=wandb.config.album_shift_limit,
    album_scale_limit=wandb.config.album_scale_limit,
    album_rotate_limit=wandb.config.album_rotate_limit,
    p_album_shiftscalerotate=wandb.config.p_album_shiftscalerotate,
    p_flip=wandb.config.p_flip,
    channelwisedropout_max_fraction=wandb.config.channelwisedropout_max_fraction,
    channelwisedropout_min_width=wandb.config.channelwisedropout_min_width,
    channelwisedropout_min_height=wandb.config.channelwisedropout_min_height,
    channelwisedropout_max_holes=wandb.config.channelwisedropout_max_holes,
    p_channelwisedropout=wandb.config.p_channelwisedropout,
)
