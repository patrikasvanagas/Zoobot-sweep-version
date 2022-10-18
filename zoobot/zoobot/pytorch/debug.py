import logging
import pandas as pd
import os
from pytorch_lightning.plugins.training_type import DDPPlugin
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_galaxy_datasets.galaxy_datamodule import GalaxyDataModule
from zoobot.pytorch.training import losses
from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.estimators import efficientnet_standard, resnet_torchvision_custom  # also resnet_detectron2_custom, imported below
import sys
sys.path.insert(0,'home/patrikas_v/to_zip/zoobot')
from zoobot.shared import label_metadata, schemas
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_galaxy_datasets.prepared_datasets import (
    DecalsDR5Dataset,
    decals_dr5_setup,
    GZ2Dataset,
    gz2_setup
)
from pytorch_galaxy_datasets.galaxy_dataset import GalaxyDataset
from zoobot.pytorch.training import train_with_pytorch_lightning

from PIL import Image
from simplejpeg import decode_jpeg

import sys
from typing import Optional
import logging

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_galaxy_datasets import galaxy_dataset

sys.path.insert(0,'/home/patrikas_v/to_zip/AstroAugmentations')
import astroaugmentations as AA
from astroaugmentations.datasets.galaxy_mnist import GalaxyMNIST
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import simplejpeg


import torch
import os
import logging

import numpy as np
import pandas as pd

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s %(levelname)s: %(message)s'
# )
# logging.info('Begin training on catalog example script')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# question_answer_pairs = label_metadata.decals_dr5_ortho_pairs
# dependencies = label_metadata.decals_ortho_dependencies
# schema = schemas.Schema(question_answer_pairs, dependencies)

# logging.info('Schema: {}'.format(schema))

train_catalog, train_label_cols = decals_dr5_setup(
    root='/home/patrikas_v/to_zip/decals',  
    train=True,
    download=False
)
val_catalog, test_label_cols = decals_dr5_setup(
    root='/home/patrikas_v/to_zip/decals', 
    train=False,
    download=False
)

# adjusted_catalog = train_catalog.sample(1000)

galaxy = train_catalog.iloc[1]

# galaxy['file_loc']

def get_galaxy_label(galaxy, label_cols):
    # no longer casts to int64, user now responsible in df. If dtype is mixed, will try to infer with infer_objects
    return galaxy[label_cols].infer_objects().values.squeeze()  # squeeze for if there's one label_col

with open(galaxy['file_loc'], 'rb') as f:
    og_image = Image.fromarray(decode_jpeg(f.read()))

label = get_galaxy_label(galaxy,train_label_cols) #PIL!

og_image


class ToGray():

    def __init__(self, reduce_channels=False):
        if reduce_channels:
            self.mean = lambda arr: arr.mean(axis=2, keepdims=True)
        else:
            self.mean = lambda arr: arr.mean(
                axis=2, keepdims=True).repeat(3, axis=2)

    def __call__(self, image, **kwargs):
        return self.mean(image)

class GrayscaleUnweighted(torch.nn.Module):

    def __init__(self, num_output_channels=1):
        super().__init__()
        self.num_output_channels = num_output_channels

    def forward(self, img):
        """
        PyTorch (and tensorflow) does greyscale conversion as a *weighted* mean by default (as colours have different perceptual brightnesses).
        Here, do a simple mean.
        Args:
            img (Tensor): Image to be converted to grayscale.

        Returns:
            Tensor: Grayscaled image.
        """
        # https://pytorch.org/docs/stable/generated/torch.mean.html
        return img.mean(dim=-3, keepdim=True)  # (..., C, H, W) convention

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)

def default_torchvision_transforms(greyscale, resize_size, crop_scale_bounds, crop_ratio_bounds):
    # refactored out for use elsewhere, if need exactly these transforms
    # assume input is 0-255 uint8 tensor

    # automatically normalises from 0-255 int to 0-1 float
    transforms_to_apply = [transforms.ToTensor()]  # dataset gives PIL image currently

    if greyscale:
        # transforms.Grayscale() adds perceptual weighting to rgb channels
        transforms_to_apply += [GrayscaleUnweighted()]

    transforms_to_apply += [
        transforms.RandomResizedCrop(
            size=resize_size,  # assumed square
            scale=crop_scale_bounds,  # crop factor
            ratio=crop_ratio_bounds,  # crop aspect ratio
            interpolation=transforms.InterpolationMode.BILINEAR),  # new aspect ratio
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(
            degrees=180., interpolation=transforms.InterpolationMode.BILINEAR)
    ]
    
    return transforms_to_apply

def transform_with_astroaugmentations(greyscale):
    if greyscale:
        transforms_to_apply = [
            A.Lambda(
                name="ToGray", image=ToGray(reduce_channels=True), always_apply=True
            )
        ]
        # transforms_to_apply = [A.ToGray()]
    else:
        transforms_to_apply = []
        
    transforms_to_apply += (
        [
            A.ToFloat(max_value=255),
            A.Lambda(
                name="Adding Simulated Sources",
                image=AA.optical.SuperimposeSources(mode="sersic"),
                p=1,
            ),
            ToTensorV2(),
        ]
    )
    astroaugmentations_transform = A.Compose(transforms_to_apply)
    return astroaugmentations_transform


astroaugmentations_transform = transform_with_astroaugmentations(greyscale=1)

astroaugmentations_img = astroaugmentations_transform(image = np.array(og_image))["image"]

plt.imshow(astroaugmentations_img.permute(1, 2, 0))