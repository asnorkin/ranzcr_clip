import os
import os.path as osp

import albumentations as A
import cv2 as cv
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


def read_dirs(dirpath: str) -> list:
    return [osp.join(dirpath, dirname) for dirname in os.listdir(dirpath) if osp.isdir(dirname)]


class ImageItemsDataset(Dataset):
    def __init__(self, items: list, classes: list, transform: A.BasicTransform = None, indices: list = None):
        self.items = items
        self.classes = classes
        self.transform = transform
        self.indices = indices or list(range(len(items)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        sample = self._load_sample(index)

        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def setup_indices(self, indices: list) -> None:
        self.indices = indices

    def _index(self, index: int) -> int:
        return self.indices[index]

    def _load_sample(self, index: int) -> dict:
        raise NotImplementedError

    @classmethod
    def load_image(cls, image_file: str, fmt: str = 'gray') -> np.ndarray:
        if fmt == 'gray':
            image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        elif fmt == 'rgb':
            image = cv.imread(image_file, cv.IMREAD_COLOR)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        else:
            raise ValueError(f'Unsupported image format: {fmt}. Supported are: gray, rgb')

        return image


class XRayDataset(ImageItemsDataset):
    @property
    def targets(self) -> list:
        return [item['target'] for item in self.items]

    def _load_sample(self, index: int) -> dict:
        index = self._index(index)
        item = self.items[index]

        sample = {
            'image': self.load_image(item['image_file']),
            'target': item['target'],
            'index': index,
        }

        return sample

    @classmethod
    def load_items(cls, labels_csv: str, images_dir: str):
        # Read labels
        labels_df = pd.read_csv(labels_csv)
        classes = list(labels_df.columns[1:-2])

        # Fix items order
        labels_df.sort_values(by='StudyInstanceUID', inplace=True)

        # Load items
        items, not_found = [], 0
        for _, row in tqdm(labels_df.iterrows(), desc='Loading items', unit='item'):
            image_file = osp.join(images_dir, f'{row.StudyInstanceUID}.jpg')
            if not osp.exists(image_file):
                not_found += 1
                continue

            items.append(
                {
                    'instance_uid': row.StudyInstanceUID,
                    'image_file': image_file,
                    'target': row[classes].values.astype(np.float),
                    'patient_id': row['PatientID'],
                    'fold': row.fold,
                }
            )

        # Add indices to items after loading because some images can be skipped
        for i, _ in enumerate(items):
            items[i]['index'] = i

        print('Dataset successfully loaded.')
        if not_found > 0:
            print(f'Not found {not_found} images out of. They was skipped.')

        return items, classes

    @classmethod
    def create(cls, labels_csv: str, images_dir: str, transform: A.BasicTransform = None):
        items, classes = cls.load_items(labels_csv, images_dir)
        return cls(items, classes, transform=transform)


class InferenceXRayDataset(ImageItemsDataset):
    def _load_sample(self, index: int) -> dict:
        item = self.items[self._index(index)]

        sample = {
            'image': item['image'],
            'instance_uid': item['instance_uid'],
        }

        if sample['image'] is None:
            sample['image'] = self.load_image(item['image_file'])

        return sample

    @classmethod
    def load_items(cls, images_dir: str) -> list:
        image_files = [osp.join(images_dir, fname) for fname in os.listdir(images_dir)]

        items = []
        for image_file in tqdm(image_files, desc='Loading dataset', unit='image'):
            items.append(
                {
                    'image': None,
                    'image_file': image_file,
                    'instance_uid': osp.splitext(osp.basename(image_file))[0],
                }
            )

        return items

    @classmethod
    def create(cls, images_dir: str, transform: A.BasicTransform = None):
        items = cls.load_items(images_dir)
        return cls(items, classes=[], transform=transform)


if __name__ == '__main__':
    dataset = XRayDataset.create(labels_csv='data/train.csv', images_dir='data/train')
    print('XRayDataset successfully created!')
