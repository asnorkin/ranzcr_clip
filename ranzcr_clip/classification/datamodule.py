from argparse import ArgumentParser, Namespace
from math import floor
from typing import Optional

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader, Sampler

from classification.dataset import XRayDataset
from classification.modelzoo import ModelConfig


class XRayClassificationDataModule(pl.LightningDataModule):
    def __init__(self, hparams: Namespace, items: Optional[list] = None, classes: Optional[list] = None):
        super().__init__()

        self.config = ModelConfig(hparams.config_file)
        self.hparams = hparams

        # Batch size that can be changed
        self.batch_size = self.hparams.batch_size

        # Common placeholders
        self.items = items
        self.classes = classes
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None) -> None:
        # Skip setup on test stage
        if stage != 'train' and self.train_dataset is not None:
            return

        # Load and split items
        if self.items is None or self.classes is None:
            self.items, self.classes = XRayDataset.load_items(
                labels_csv=self.hparams.labels_csv, images_dir=self.hparams.images_dir
            )

        # Set up fold and split items for this fold
        fold = self.hparams.fold
        train_indices = [i for i, item in enumerate(self.items) if item['fold'] != fold]
        val_indices = [i for i, item in enumerate(self.items) if item['fold'] == fold]

        # Transforms
        pre_transforms = []

        augmentations = [
            A.RandomResizedCrop(height=self.config.input_height, width=self.config.input_width, scale=(0.9, 1), p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
            A.OneOf(
                [
                    A.OpticalDistortion(distort_limit=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=1.0),
                    A.ElasticTransform(alpha=3),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(),
                    A.MotionBlur(),
                    A.MedianBlur(),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.ImageCompression(),
                    A.Downscale(scale_min=0.1, scale_max=0.15),
                ],
                p=0.2,
            ),
            A.IAAPiecewiseAffine(p=0.2),
            A.IAASharpen(p=0.2),
            A.CoarseDropout(
                max_height=int(self.config.input_height * 0.1),
                max_width=int(self.config.input_width * 0.1),
                max_holes=5,
                p=0.5,
            ),
        ]

        post_transforms = [
            A.Resize(height=self.config.input_height, width=self.config.input_width),
            A.CLAHE(always_apply=True),
            A.Normalize(mean=0.482, std=0.220, always_apply=True),  # Ranzcr
            ToTensorV2(always_apply=True),
        ]

        # Train dataset
        train_transform = A.Compose(pre_transforms + augmentations + post_transforms)
        self.train_dataset = XRayDataset(
            items=self.items,
            classes=self.classes,
            transform=train_transform,
            indices=train_indices,
        )

        # Val dataset
        val_transform = A.Compose(pre_transforms + post_transforms)
        self.val_dataset = XRayDataset(
            items=self.items,
            classes=self.classes,
            transform=val_transform,
            indices=val_indices,
        )

    def setup_input_size(self, input_size: int) -> None:
        current_input_size = self.train_dataset.transforms[0].height
        current_batch_size = self.batch_size
        new_batch_size = floor((input_size / current_input_size) ** 2 * current_batch_size)
        if new_batch_size < 1:
            print(f'Can\'t set new input size {input_size} because new batch size will be bad: {new_batch_size}')
            return

        self.batch_size = new_batch_size

        self.train_dataset.transforms[0].height = input_size
        self.train_dataset.transforms[0].width = input_size
        self.val_dataset.transforms[0].height = input_size
        self.val_dataset.transforms[0].width = input_size

        print(f'Successfully set up nwe input size {input_size} with new batch size {new_batch_size}')

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def _dataloader(
        self, dataset: XRayDataset, sampler: Optional[Sampler] = None, shuffle: bool = False
    ) -> DataLoader:
        params = {
            'drop_last': False,
            'pin_memory': True,
            'batch_size': self.batch_size,
            'num_workers': self.hparams.num_workers,
        }

        if sampler is not None:
            params['sampler'] = sampler
        else:
            params['shuffle'] = shuffle

        return DataLoader(dataset, **params)

    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Paths
        parser.add_argument('--labels_csv', type=str, default='data/train.csv')
        parser.add_argument('--images_dir', type=str, default='data/train')

        # General
        parser.add_argument('--num_epochs', type=int, default=20)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=32)

        return parser
