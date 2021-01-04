from argparse import ArgumentParser
from math import floor

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import pytorch_lightning as pl
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data.dataloader import DataLoader

from classification.dataset import XRayDataset
from classification.modelzoo import ModelConfig


def grouped_train_test_split(items, groups=None, test_size=0.2, random_state=None):
    gss = GroupShuffleSplit(n_splits=2, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(gss.split(X=items, y=None, groups=groups))
    return [items[i] for i in train_indices], [items[i] for i in test_indices]


class XRayClassificationDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()

        self.config = ModelConfig(hparams.config_file)
        self.hparams = hparams

        # Batch size that can be changed
        self.batch_size = self.hparams.batch_size

        # Common placeholders
        self.items = None
        self.classes = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None

        # Cross validation placeholders
        self.cv = None
        self.train_indices = None
        self.val_indices = None

    def setup(self, stage=None):
        # Skip setup on test stage
        if stage != 'train' and self.train_dataset is not None:
            return

        # Load and split items
        self.items, self.classes, images = XRayDataset.load_items(
            labels_csv=self.hparams.labels_csv,
            images_dir=self.hparams.images_dir,
            num_workers=self.hparams.num_workers,
            cache_images=self.hparams.cache_images,
            cache_size=(self.config.input_width, self.config.input_height))

        patient_ids = [item['patient_id'] for item in self.items]
        targets = [item['target'] for item in self.items]

        if self.hparams.cv_folds is not None:
            self.cv = GroupKFold(n_splits=self.hparams.cv_folds)

            # GroupKFold has no stratification by y
            # y=targets only for convenience
            self.train_indices, self.val_indices = [], []
            for fold_train_indices, fold_val_indices in self.cv.split(self.items, y=targets, groups=patient_ids):
                self.train_indices.append(fold_train_indices)
                self.val_indices.append(fold_val_indices)

            train_items = val_items = self.items

        elif self.hparams.val_size is None:
            train_items = val_items = self.items
        else:
            train_items, val_items = \
                grouped_train_test_split(
                    self.items,
                    groups=patient_ids,
                    test_size=self.hparams.val_size,
                    random_state=self.hparams.seed)

        # Transforms
        pre_transforms = []
        if not self.hparams.cache_images:
            pre_transforms.append(A.Resize(height=self.config.input_height, width=self.config.input_width))

        augmentations = [
            A.RandomResizedCrop(height=self.config.input_height, width=self.config.input_width, scale=(0.85, 1.0)),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.Rotate(limit=3),
            A.CoarseDropout(),
        ]

        post_transforms = [
            A.FromFloat('uint8', always_apply=True),
            A.CLAHE(always_apply=True),
            # A.Normalize(mean=0.449, std=0.226, always_apply=True),    # ImageNet
            A.Normalize(mean=0.482, std=0.220, always_apply=True),    # Ranzcr
            ToTensorV2(always_apply=True),
        ]

        # Train dataset
        train_transform = A.Compose(pre_transforms + augmentations + post_transforms)
        self.train_dataset = XRayDataset(items=train_items, classes=self.classes,
                                         transform=train_transform, images=images)

        # Val dataset
        val_transform = A.Compose(pre_transforms + post_transforms)
        self.val_dataset = XRayDataset(items=val_items, classes=self.classes,
                                       transform=val_transform, images=images)

    def setup_fold(self, fold):
        self.train_dataset.setup_indices(self.train_indices[fold])
        self.val_dataset.setup_indices(self.val_indices[fold])

    def setup_input_size(self, input_size):
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

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def _dataloader(self, dataset, sampler=None, shuffle=False):
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
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Paths
        parser.add_argument('--labels_csv', type=str, default='data/train.csv')
        parser.add_argument('--images_dir', type=str, default='data/train')

        # General
        parser.add_argument('--num_epochs', type=int, default=20)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--cv_folds', type=int, default=None)
        parser.add_argument('--val_size', type=float, default=None,
                            help='val_size=None means use all train set without augmentations for validation')
        parser.add_argument('--cache_images', action='store_true')

        return parser
