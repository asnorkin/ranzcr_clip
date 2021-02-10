import albumentations as A
import cv2 as cv
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset


def load_train_labels(train_labels_file: str) -> pd.DataFrame:
    labels_df = pd.read_csv(train_labels_file)

    # Fix errors
    labels_df.loc[
        labels_df.StudyInstanceUID == '1.2.826.0.1.3680043.8.498.93345761486297843389996628528592497280',
        'ETT - Abnormal',
    ] = 0
    labels_df.loc[
        labels_df.StudyInstanceUID == '1.2.826.0.1.3680043.8.498.93345761486297843389996628528592497280',
        'CVC - Abnormal',
    ] = 1

    # Fix items order
    labels_df.sort_values(by='StudyInstanceUID', inplace=True)

    return labels_df


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

        sample = self._postprocess_sample(sample)

        return sample

    def setup_indices(self, indices: list) -> None:
        self.indices = indices

    def _index(self, index: int) -> int:
        return self.indices[index]

    def _load_sample(self, index: int) -> dict:
        raise NotImplementedError

    @staticmethod
    def _postprocess_sample(sample: dict) -> dict:
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
