import os
import os.path as osp

import cv2 as cv
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm


def read_dirs(dirpath):
    return [osp.join(dirpath, dirname) for dirname in os.listdir(dirpath) if osp.isdir(dirname)]


class XRayDataset(Dataset):
    def __init__(self, items, classes, transform=None, items_per_epoch=None):
        self.items = items
        self.classes = classes
        self.transform = transform
        self.items_per_epoch = items_per_epoch or len(self.items)
        self.buckets = len(self.items) // self.items_per_epoch + int(len(self.items) % self.items_per_epoch)

    def __len__(self):
        return self.items_per_epoch

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        sample = self._load_sample(index)

        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def _load_sample(self, index):
        if self.buckets > 1:
            bucket_offset = np.random.randint(0, self.buckets) * self.items_per_epoch
            index = bucket_offset + index
            index = index % len(self.items)

        item = self.items[index]

        sample = {
            'image': item['image'],
            'target': item['target'],
        }

        if sample['image'] is None:
            sample['image'] = self.load_image(item['image_file'])

        return sample

    @classmethod
    def load_items(cls, labels_csv, images_dir, cache_images=False):
        items = []

        labels_df = pd.read_csv(labels_csv)
        classes = list(labels_df.columns[1:-1])

        with tqdm(desc='Loading dataset', unit='image', total=len(labels_df)) as progress_bar:
            for i, row in labels_df.iterrows():
                image_file = osp.join(images_dir, f'{row.StudyInstanceUID}.jpg')
                if not osp.exists(image_file):
                    continue

                items.append({
                    'image_file': image_file,
                    'image': cls.load_image(image_file) if cache_images else None,
                    'target': row[classes].values.astype(np.float),
                })
                progress_bar.update()

        return items, classes

    @classmethod
    def create(cls, labels_csv, images_dir, cache_images=False, transform=None, items_per_epoch=None):
        items, classes = cls.load_items(labels_csv, images_dir, cache_images=cache_images)
        return cls(items, classes, transform=transform, items_per_epoch=items_per_epoch)

    @classmethod
    def load_image(cls, image_file):
        image = cv.imread(image_file, cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.
        return image


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
