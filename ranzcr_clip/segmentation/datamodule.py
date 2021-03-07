from argparse import ArgumentParser, Namespace
from math import floor
from typing import Optional

import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler

from common.model_utils import ModelConfig
from segmentation.dataset import load_items, XRayCatheterDataset, XRayLungDataset


class LungSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, hparams: Namespace, items: Optional[list] = None):
        super().__init__()

        self.config = ModelConfig(hparams.config_file)
        self.hparams = hparams

        # Batch size that can be changed
        self.batch_size = self.hparams.batch_size
        self.input_size = self.config.input_size

        # Common placeholders
        self.items = items
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None) -> None:
        # Skip setup on test stage
        if stage != 'train' and self.train_dataset is not None:
            return

        # Load and split items
        if self.items is None:
            self.items = load_items(
                project=self.hparams.project,
                images_dir=self.hparams.images_dir,
                labels_csv=self.hparams.labels_csv,
                masks_dir=self.hparams.masks_dir,
                annotations_csv=self.hparams.annotations_csv,
            )

        # Set up fold and split items for this fold
        fold = self.hparams.fold
        train_indices = [i for i, item in enumerate(self.items) if item['fold'] != fold]
        val_indices = [i for i, item in enumerate(self.items) if item['fold'] == fold]

        # Transforms
        pre_transforms = []

        augmentations = [
            A.RandomResizedCrop(height=self.input_size, width=self.input_size, scale=(0.9, 1), p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7),
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
                max_height=int(self.input_size * 0.1),
                max_width=int(self.input_size * 0.1),
                max_holes=5,
                p=0.5,
            ),
        ]

        post_transforms = [
            A.Resize(height=self.input_size, width=self.input_size),
            A.CLAHE(always_apply=True),
            A.Normalize(mean=0.482, std=0.220, always_apply=True),  # Ranzcr
            ToTensorV2(always_apply=True),
        ]

        dataset_class = XRayLungDataset if 'lung' in self.hparams.project else XRayCatheterDataset

        # Train dataset
        train_transform = A.Compose(pre_transforms + augmentations + post_transforms)
        self.train_dataset = dataset_class(
            items=self.items,
            classes=[],
            transform=train_transform,
            indices=train_indices,
        )

        # Val dataset
        val_transform = A.Compose(pre_transforms + post_transforms)
        self.val_dataset = dataset_class(
            items=self.items,
            classes=[],
            transform=val_transform,
            indices=val_indices,
        )

    def setup_input_size(self, input_size: int) -> None:
        current_input_size = self.train_dataset.transform[0].height
        current_batch_size = self.batch_size
        new_batch_size = floor((current_input_size / input_size) ** 2 * current_batch_size)
        if new_batch_size < 1:
            print(f'Can\'t set new input size {input_size} because new batch size will be bad: {new_batch_size}')
            return

        self.batch_size = new_batch_size
        self.input_size = input_size

        self.train_dataset.transform[0].height = input_size
        self.train_dataset.transform[0].width = input_size
        self.val_dataset.transform[0].height = input_size
        self.val_dataset.transform[0].width = input_size

        self.trainer.reset_train_dataloader(self.trainer.get_model())
        self.trainer.reset_val_dataloader(self.trainer.get_model())

        print(f'Successfully set up nwe input size {input_size} with new batch size {new_batch_size}')

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def _dataloader(
        self, dataset: XRayLungDataset, sampler: Optional[Sampler] = None, shuffle: bool = False
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
        parser.add_argument('--annotations_csv', type=str, default='data/train_annotations.csv')
        parser.add_argument('--images_dir', type=str, default='data/train')
        parser.add_argument('--masks_dir', type=str, default='data/train_lung_masks')

        # General
        parser.add_argument('--num_epochs', type=int, default=20)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=32)

        return parser
