from glob import glob
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

from pytorch_galaxy_datasets.prepared_datasets import (
    DecalsDR5Dataset,
    decals_dr5_setup,
    GZ2Dataset,
    gz2_setup
)
sys.path.insert(0,'/home/patrikas_v/to_zip/AstroAugmentations/astroaugmentations/')
import image_domain

# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
class GalaxyDataModule(pl.LightningDataModule):
    # takes generic catalogs (which are already downloaded and happy),
    # splits if needed, and creates generic datasets->dataloaders etc
    # easy to make dataset-specific default transforms if desired
    def __init__(
        self,
        label_cols,
        # provide full catalog for automatic split, or...
        catalog=None,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        # provide train/val/test catalogs for your own previous split
        train_catalog=None,
        val_catalog=None,
        test_catalog=None,
        predict_catalog=None,
        # augmentation params (sensible supervised defaults)
        greyscale=True,
        album=False,
        astroaug=False,

        crop_scale_bounds=(0.7, 0.8),
        crop_ratio_bounds=(0.9, 1.1),
        # hardware params
        batch_size=256,  # careful - will affect final performance
        use_memory=False,  # deprecated
        num_workers=4,
        prefetch_factor=4,
        seed=42,

        resize_size=224,        

        elastic_sigma_datamodule = 100,
        elastic_alpha_affine_datamodule = 2,
        p_elastic_datamodule = 0.5,

        sersic_extent_min_datamodule = 5,
        sersic_extent_max_datamodule = 80,
        sersic_max_number_datamodule = 5,
        p_sersic_datamodule = 0.5,

        gaussian_extent_min_datamodule = 3,
        gaussian_extent_max_datamodule = 20,
        gaussian_max_number_datamodule = 5,
        p_gaussian_datamodule = 0.5,

        p_croppedtemplateoverlap_datamodule = 0.5,

        brightness_gradient_minimum_datamodule = 0.5,
        brightness_gradient_noise_datamodule = 0.2,
        p_brightness_gradient_datamodule = 0.5,

        album_shift_limit_datamodule = 0.1,
        album_scale_limit_datamodule = 0.1,
        album_rotate_limit_datamodule = 180,
        p_album_shiftscalerotate_datamodule = 0.5,

        p_flip_datamodule = 0.5,

        channelwisedropout_max_fraction_datamodule = 0.2,
        channelwisedropout_min_width_datamodule = 10,
        channelwisedropout_min_height_datamodule = 10,
        channelwisedropout_max_holes_datamodule = 0.5,
        p_channelwisedropout_datamodule = 0.5,

    ):
        super().__init__()

        if catalog is not None:  # catalog provided, should not also provide explicit split catalogs
            assert train_catalog is None
            assert val_catalog is None
            assert test_catalog is None
        else:  # catalog not provided, must provide explicit split catalogs - at least one
            assert (train_catalog is not None) or (val_catalog is not None) or (test_catalog is not None) or (predict_catalog is not None)
            # see setup() for how having only some explicit catalogs is handled

        self.label_cols = label_cols
        self.catalog = catalog
        self.train_catalog = train_catalog
        self.val_catalog = val_catalog
        self.test_catalog = test_catalog
        self.predict_catalog = predict_catalog
        self.batch_size = batch_size
        self.resize_size = resize_size
        self.crop_scale_bounds = crop_scale_bounds
        self.crop_ratio_bounds = crop_ratio_bounds
        self.use_memory = use_memory
        if self.use_memory:
            raise NotImplementedError
        self.num_workers = num_workers
        self.seed = seed
        assert np.isclose(train_fraction + val_fraction + test_fraction, 1.)
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.greyscale = greyscale
        self.album = album
        self.astroaug = astroaug
        self.elastic_sigma_datamodule = elastic_sigma_datamodule
        self.elastic_alpha_affine_datamodule = elastic_alpha_affine_datamodule
        self.p_elastic_datamodule = p_elastic_datamodule
        self.sersic_extent_min_datamodule = sersic_extent_min_datamodule,
        self.sersic_extent_max_datamodule = sersic_extent_max_datamodule
        self.sersic_max_number_datamodule = sersic_max_number_datamodule
        self.p_sersic_datamodule = p_sersic_datamodule
        self.gaussian_extent_min_datamodule = gaussian_extent_min_datamodule
        self.gaussian_extent_max_datamodule = gaussian_extent_max_datamodule
        self.gaussian_max_number_datamodule = gaussian_max_number_datamodule
        self.p_gaussian_datamodule = p_gaussian_datamodule
        self.p_croppedtemplateoverlap_datamodule = p_croppedtemplateoverlap_datamodule
        self.brightness_gradient_minimum_datamodule = brightness_gradient_minimum_datamodule
        self.brightness_gradient_noise_datamodule = brightness_gradient_noise_datamodule
        self.p_brightness_gradient_datamodule = p_brightness_gradient_datamodule
        self.album_shift_limit_datamodule = album_shift_limit_datamodule
        self.album_scale_limit_datamodule = album_scale_limit_datamodule
        self.album_rotate_limit_datamodule = album_rotate_limit_datamodule
        self.p_album_shiftscalerotate_datamodule = p_album_shiftscalerotate_datamodule
        self.p_flip_datamodule = p_flip_datamodule
        self.channelwisedropout_max_fraction_datamodule = channelwisedropout_max_fraction_datamodule
        self.channelwisedropout_min_width_datamodule = channelwisedropout_min_width_datamodule
        self.channelwisedropout_min_height_datamodule = channelwisedropout_min_height_datamodule
        self.channelwisedropout_max_holes_datamodule = channelwisedropout_max_holes_datamodule
        self.p_channelwisedropout_datamodule = p_channelwisedropout_datamodule

        if self.album:
            logging.info('Using albumentations for augmentations')
            self.transform_with_album()
        elif self.astroaug:
            logging.info('Using astroaugmentations for augmentations')
            self.transform_with_astroaugmentations()
        elif not self.album and not self.astroaug:
        # else:
            logging.info('Using torchvision for augmentations')
            self.transform_with_torchvision()
        self.prefetch_factor = prefetch_factor
        self.dataloader_timeout = 240  # seconds
        logging.info('Num workers: {}'.format(self.num_workers))
        logging.info('Prefetch factor: {}'.format(self.prefetch_factor))

    def transform_with_astroaugmentations(self):
        if self.greyscale:
            transforms_to_apply = [A.Lambda(name='ToGray', image=ToGray(
                reduce_channels=True), always_apply=True)]
        else:
            transforms_to_apply = [A.ToFloat()]

        transforms_to_apply += (
            [
                A.LongestMaxSize(
                    max_size = self.resize_size
                ),

                A.ElasticTransform(
                    alpha=1, sigma=self.elastic_sigma_datamodule, alpha_affine=self.elastic_alpha_affine_datamodule, interpolation=1,
                    border_mode=1, value=0,
                    p=self.p_elastic_datamodule
                ),

                A.Lambda(
                    name='AddGaussianSources',
                    image=image_domain.optical.SuperimposeSources(
                        mode='gaussian',
                        max_number=self.gaussian_max_number_datamodule,
                        extent=(self.gaussian_extent_min_datamodule, self.gaussian_extent_max_datamodule),
                        scaling=None
                    ),
                    p=self.p_gaussian_datamodule
                ),

                A.Lambda(
                    name='AddingRealData',
                    image=image_domain.optical.CroppedTemplateOverlap(
                        mode='catalog', catalog=self.train_catalog, resolution = self.resize_size,
                    ),
                    p=self.p_croppedtemplateoverlap_datamodule
                ),

                A.Lambda(
                    name="Brightness perspective distortion", 
                    image=image_domain.BrightnessGradient(
                        limits=[self.brightness_gradient_minimum_datamodule, 1],
                        noise = self.brightness_gradient_noise_datamodule),
                    p=self.p_brightness_gradient_datamodule
                ),

                A.ShiftScaleRotate(
                    shift_limit=self.album_shift_limit_datamodule, scale_limit=self.album_scale_limit_datamodule,
                    rotate_limit=self.album_rotate_limit_datamodule, interpolation=2,
                    border_mode=0,
                p=self.p_album_shiftscalerotate_datamodule),

                A.Flip(p=self.p_flip_datamodule),

                A.Lambda(
                    name="MissingData",
                    image=image_domain.optical.ChannelWiseDropout(
                        max_fraction=self.channelwisedropout_max_fraction_datamodule,
                        min_width=self.channelwisedropout_min_width_datamodule,
                        min_height=self.channelwisedropout_min_height_datamodule,
                        max_holes=self.channelwisedropout_max_holes_datamodule,
                        channelwise_application=True,
                    ),
                    p=self.p_channelwisedropout_datamodule
                ),

                ToTensorV2(),
            ]
        )
        astroaugmentations_transform = A.Compose(transforms_to_apply)
        self.transform = lambda img: astroaugmentations_transform(image=np.array(img))["image"]

    def transform_with_torchvision(self):

        transforms_to_apply = default_torchvision_transforms(self.greyscale, self.resize_size, self.crop_scale_bounds, self.crop_ratio_bounds)

        self.transform = transforms.Compose(transforms_to_apply)
    def transform_with_album(self):
        if self.greyscale:
            transforms_to_apply = [A.Lambda(name='ToGray', image=ToGray(
                reduce_channels=True), always_apply=True)]
        else:
            transforms_to_apply = [A.ToFloat()]
        transforms_to_apply += [A.Rotate(limit=180, interpolation=1,
                        always_apply=True, border_mode=0, value=0),
            A.RandomResizedCrop(
                height=self.resize_size,  # after crop resize
                width=self.resize_size,
                scale=self.crop_scale_bounds,  # crop factor
                ratio=self.crop_ratio_bounds,  # crop aspect ratio
                interpolation=1,  # This is "INTER_LINEAR" == BILINEAR interpolation. See: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
                always_apply=True
            ),  # new aspect ratio
            A.VerticalFlip(p=0.5),
            ToTensorV2()
        ]

        albumentations_transform = A.Compose(transforms_to_apply)

        # warning - might need a transpose check
        # albumentations expects np array, and returns dict keyed by "image"
        self.transform = lambda img: albumentations_transform(image=np.array(img))["image"]

    # only called on main process
    def prepare_data(self):
        pass   # could include some basic checks

    # called on every gpu

    def setup(self, stage: Optional[str] = None):

        if self.catalog is not None:
            # will split the catalog into train, val, test here
            self.train_catalog, hidden_catalog = train_test_split(
                self.catalog, train_size=self.train_fraction, random_state=self.seed
            )
            self.val_catalog, self.test_catalog = train_test_split(
                hidden_catalog, train_size=self.val_fraction/(self.val_fraction + self.test_fraction), random_state=self.seed
            )
            del hidden_catalog
        else:
            # assume you have passed pre-split catalogs
            # (maybe not all, e.g. only a test catalog, or only train/val catalogs)
            if stage == 'predict':
                assert self.predict_catalog is not None
            elif stage == 'test':
                # only need test
                assert self.test_catalog is not None
            elif stage == 'fit':
                # only need train and val
                assert self.train_catalog is not None
                assert self.val_catalog is not None
            else:
                # need all three (predict is still optional)
                assert self.train_catalog is not None
                assert self.val_catalog is not None
                assert self.test_catalog is not None
            # (could write this shorter but this is clearest)


        # Assign train/val datasets for use in dataloaders
        # assumes dataset_class has these standard args
        if stage == "fit" or stage is None:
            self.train_dataset = galaxy_dataset.GalaxyDataset(
                catalog=self.train_catalog, label_cols=self.label_cols, transform=self.transform
            )
            self.val_dataset = galaxy_dataset.GalaxyDataset(
                catalog=self.val_catalog, label_cols=self.label_cols, transform=self.transform
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = galaxy_dataset.GalaxyDataset(
                catalog=self.test_catalog, label_cols=self.label_cols, transform=self.transform
            )

        if stage == 'predict':  # not set up by default with stage=None, only if explicitly requested
            self.predict_dataset = galaxy_dataset.GalaxyDataset(
                catalog=self.predict_catalog, label_cols=self.label_cols, transform=self.transform
            )

    # def collate_fn(batch):
    #     images

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0, prefetch_factor=self.prefetch_factor, timeout=self.dataloader_timeout)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0, prefetch_factor=self.prefetch_factor, timeout=self.dataloader_timeout)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0, prefetch_factor=self.prefetch_factor, timeout=self.dataloader_timeout)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0, prefetch_factor=self.prefetch_factor, timeout=self.dataloader_timeout)


def default_torchvision_transforms(greyscale, resize_size, crop_scale_bounds, crop_ratio_bounds):
    # refactored out for use elsewhere, if need exactly these transforms
    # assume input is 0-255 uint8 tensor

    # automatically normalises from 0-255 int to 0-1 float
    transforms_to_apply = [transforms.ToTensor()]  # dataset gives PIL image currently

    if greyscale:
        # transforms.Grayscale() adds perceptual weighting to rgb channels
        transforms_to_apply += [GrayscaleUnweighted()]


    transforms_to_apply += [transforms.Resize(size = resize_size)]
    # transforms_to_apply += [
    #     transforms.RandomResizedCrop(
    #         size=resize_size,  # assumed square
    #         scale=crop_scale_bounds,  # crop factor
    #         ratio=crop_ratio_bounds,  # crop aspect ratio
    #         interpolation=transforms.InterpolationMode.BILINEAR),  # new aspect ratio
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(
    #         degrees=180., interpolation=transforms.InterpolationMode.BILINEAR)
    # ]
    
    return transforms_to_apply

# torchvision
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


# albumentations versuib of GrayscaleUnweighted
class ToGray():

    def __init__(self, reduce_channels=False):
        if reduce_channels:
            self.mean = lambda arr: arr.mean(axis=2, keepdims=True)
        else:
            self.mean = lambda arr: arr.mean(
                axis=2, keepdims=True).repeat(3, axis=2)

    def __call__(self, image, **kwargs):
        mean = self.mean(image)
        mean = A.augmentations.functional.to_float(mean, max_value=mean.max())
        return mean
        
