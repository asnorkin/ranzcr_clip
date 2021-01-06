import ctypes
import os
import os.path as osp
from multiprocessing import Array
from queue import Queue
from threading import Thread

import cv2 as cv
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm


def read_dirs(dirpath):
    return [osp.join(dirpath, dirname) for dirname in os.listdir(dirpath) if osp.isdir(dirname)]


def load_items(thread_id, results_queue, labels_df, images_dir, cache_images=False, cache_size=None):
    classes = list(labels_df.columns[1:-1])

    if isinstance(cache_size, int):
        cache_size = (cache_size, cache_size)

    items, images = [], []
    for i, row in labels_df.iterrows():
        image_file = osp.join(images_dir, f'{row.StudyInstanceUID}.jpg')
        if osp.exists(image_file):
            if cache_images:
                image = XRayDataset.load_image(image_file)
                if cache_size is not None:
                    image = cv.resize(image, cache_size)
                images.append(image)

            items.append({
                'image_file': image_file,
                'target': row[classes].values.astype(np.float),
                'patient_id': row['PatientID'],
            })

    result = {
        'thread_id': thread_id,
        'items': items,
    }

    if cache_images:
        result['images'] = images

    results_queue.put(result)


class XRayDataset(Dataset):
    def __init__(self, items, classes, transform=None, indices=None, images=None):
        self.items = items
        self.classes = classes
        self.images = images
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

    @property
    def targets(self):
        return [item['target'] for item in self.items]

    def setup_indices(self, indices):
        self.indices = indices

    def _load_sample(self, index):
        index = self._index(index)
        item = self.items[index]

        sample = {
            'image': self.images[index] if self.images is not None else None,
            'target': item['target'],
            'index': index,
        }

        if sample['image'] is None:
            sample['image'] = self.load_image(item['image_file'])

        return sample

    def _index(self, index):
        return self.indices[index]

    @classmethod
    def load_items(cls, labels_csv, images_dir, num_workers=1, cache_images=False, cache_size=None):
        # num_workers = 0 case
        num_workers = max(1, num_workers)

        # Read labels
        labels_df = pd.read_csv(labels_csv)
        classes = list(labels_df.columns[1:-1])

        print(f'Loading dataset in {num_workers} threads.')

        # Run threads
        items_per_thread = len(labels_df) // num_workers + 1
        results = Queue()
        threads = []
        for thread_id in range(num_workers):
            # Get thread part of data
            start = items_per_thread * thread_id
            finish = start + items_per_thread
            thread_df = labels_df.iloc[start: finish]

            # Create and run thread
            args = (thread_id, results, thread_df, images_dir, cache_images, cache_size)
            thread = Thread(target=load_items, args=args)
            thread.start()
            threads.append(thread)

        # Join all threads
        [t.join() for t in threads]

        # Concatenate the results into items
        items, images = [], []
        while not results.empty():
            result = results.get()
            items.extend(result['items'])
            if cache_images:
                images.extend(result['images'])

        if cache_images:
            images = np.stack(images)
            n_images, h, w = images.shape
            images = np.ctypeslib.as_array(
                Array(ctypes.c_uint, n_images * h * w).get_obj()
            ).reshape(n_images, h, w).astype('uint8')
        else:
            images = None

        print(f'Dataset successfully loaded')

        return items, classes, images

    @classmethod
    def create(cls, labels_csv, images_dir, num_workers=1, transform=None, cache_images=False, cache_size=None):
        items, classes, images = cls.load_items(labels_csv, images_dir, num_workers=num_workers,
                                                cache_images=cache_images, cache_size=cache_size)
        return cls(items, classes, transform=transform, images=images)

    @classmethod
    def load_image(cls, image_file):
        image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
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
