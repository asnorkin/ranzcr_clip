import os
import os.path as osp

import cv2 as cv
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm


def read_dirs(dirpath):
    return [osp.join(dirpath, dirname) for dirname in os.listdir(dirpath) if osp.isdir(dirname)]


def load_items(labels_df):
    pass


class XRayDataset(Dataset):
    def __init__(self, items, classes, transform=None, indices=None, cache_images=False, cache_size=None):
        self.items = items
        self.classes = classes
        self.transform = transform
        self.indices = indices or list(range(len(items)))

        self.cache_images = cache_images
        self.cache_size = cache_size
        if self.cache_size is not None and isinstance(self.cache_size, int):
            self.cache_size = (self.cache_size, self.cache_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        sample = self._load_sample(index)

        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @property
    def targets(self):
        return [item['target'] for item in self.items]

    def setup_indices(self, indices=None):
        if indices is None:
            indices = list(range(len(self.items)))
        self.indices = indices

    def _load_sample(self, index):
        index = self._index(index)
        item = self.items[index]

        sample = {
            'image': item['image'],
            'target': item['target'],
        }

        if sample['image'] is None:
            image = self.load_image(item['image_file'])

            if self.cache_images:
                if self.cache_size is not None:
                    image = cv.resize(image, self.cache_size)
                self.items[index]['image'] = image

            sample['image'] = image

        return sample

    def _index(self, index):
        return self.indices[index]

    @classmethod
    def load_items(cls, labels_csv, images_dir):
        items = []

        labels_df = pd.read_csv(labels_csv)
        classes = list(labels_df.columns[1:-1])

        with tqdm(desc='Loading dataset', unit='image', total=len(labels_df)) as progress_bar:
            for i, row in labels_df.iterrows():
                image_file = osp.join(images_dir, f'{row.StudyInstanceUID}.jpg')
                if osp.exists(image_file):
                    items.append({
                        'image_file': image_file,
                        'image': None,
                        'target': row[classes].values.astype(np.float),
                        'patient_id': row['PatientID'],
                    })

                progress_bar.update()

        return items, classes

    @classmethod
    def create(cls, labels_csv, images_dir, transform=None, cache_images=False, cache_size=None):
        items, classes = cls.load_items(labels_csv, images_dir)
        return cls(items, classes, transform=transform, cache_images=cache_images, cache_size=cache_size)

    @classmethod
    def load_image(cls, image_file):
        image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.
        return image


class InferenceXRayDataset(XRayDataset):
    def _load_sample(self, index):
        item = self.items[self._index(index)]

        sample = {
            'image': item['image'],
            'instance_uid': item['instance_uid'],
        }

        if sample['image'] is None:
            sample['image'] = self.load_image(item['image_file'])

        return sample

    @classmethod
    def load_items(cls, images_dir):
        image_files = [osp.join(images_dir, fname) for fname in os.listdir(images_dir)]

        items = []
        for image_file in tqdm(image_files, desc='Loading dataset', unit='image'):
            items.append({
                'image': None,
                'image_file': image_file,
                'instance_uid': osp.splitext(osp.basename(image_file))[0],
            })

        return items

    @classmethod
    def create(cls, images_dir, transform=None):
        items = cls.load_items(images_dir)
        return cls(items, classes=None, transform=transform)


class StratifiedLabelSampler(WeightedRandomSampler):
    def __init__(self, items, replacement=True):
        weights = self._calculate_weights(items)
        super().__init__(weights, weights.shape[0], replacement)

    @staticmethod
    def _calculate_weights(items):
        return np.ones(len(items))
        # label_counts = np.zeros_like(items[0]['target'])
        # for item in items:
        #     label_counts += item['target']
        #
        # no_label_counts = np.ones_like(label_counts) * len(items) - label_counts
        #
        # weights = np.asarray([1. / np.sqrt(label_counts[item['target']]) for item in items])
        # return weights


if __name__ == '__main__':
    labels_csv = 'data/train.csv'
    images_dir = 'data/train'
    dataset = XRayDataset.create(labels_csv='data/train.csv', images_dir='data/train')

    sampler = StratifiedLabelSampler(dataset.items)
    print(1)
