import os.path as osp
from ast import literal_eval
from typing import Any, Dict, List, Optional

import albumentations as A
import numpy as np
from PIL import Image
from scipy import interpolate
from tqdm import tqdm

from common.dataset import ImageItemsDataset, load_train_annotations, load_train_labels


class SegmentationDataset(ImageItemsDataset):
    def _load_sample(self, index: int) -> Dict[str, Any]:
        index = self._index(index)
        item = self.items[index]

        sample = {
            'image': self.load_image(item['image_file']),
            'mask': self.load_mask(item),
            'index': index,
        }

        return sample

    @classmethod
    def load_mask(cls, item):
        return cls.load_image(item['mask_file'])[..., None] / 255.0

    @staticmethod
    def _postprocess_sample(sample: dict) -> dict:
        return sample


class XRayLungDataset(SegmentationDataset):
    @classmethod
    def load_items(cls, labels_csv: str, images_dir: str, masks_dir: str) -> list:
        labels_df = load_train_labels(labels_csv)

        # Load items
        items, not_found = [], 0
        for _, row in tqdm(labels_df.iterrows(), desc='Loading items', unit='item', total=len(labels_df)):
            image_file = osp.join(images_dir, f'{row.StudyInstanceUID}.jpg')
            mask_file = osp.join(masks_dir, f'{row.StudyInstanceUID}.jpg')
            if not osp.exists(image_file) or not osp.exists(mask_file):
                not_found += 1
                continue

            items.append(
                {
                    'instance_uid': row.StudyInstanceUID,
                    'image_file': image_file,
                    'mask_file': mask_file,
                    'patient_id': row.PatientID,
                    'fold': row.fold,
                }
            )

        # Add indices to items after loading because some images can be skipped
        for i, _ in enumerate(items):
            items[i]['index'] = i

        print('Dataset successfully loaded.')
        if not_found > 0:
            n_images = labels_df.shape[0]
            print(f'Not found {not_found} images out of {n_images}. They was skipped.')

        return items

    @classmethod
    def create(cls, labels_csv: str, images_dir: str, masks_dir: str, transform: A.BasicTransform = None):
        items = cls.load_items(labels_csv, images_dir, masks_dir)
        return cls(items, classes=[], transform=transform)


def filter_duplicates(seq):
    has_diff = np.asarray([True] + (seq[1:] != seq[:-1]).any(axis=1).tolist())
    return seq[has_diff]


def interpolate_mask(points):
    if len(points) < 2:
        raise ValueError('interpolate_mask error: can\'t interpolate less than two points mask.')

    # Calculate upper bound for approximated number of points
    num_points = 0
    for i, point in enumerate(points[:-1]):
        dx = np.abs(points[i + 1, 0] - point[0])
        dy = np.abs(points[i + 1, 1] - point[1])
        num_points += dx + dy

    # Interpolate using spline
    k = 1 if len(points) - 1 < 3 else 3
    tck, *_ = interpolate.splprep([points[:, 0], points[:, 1]], s=0.0, k=k)
    xs, ys = interpolate.splev(np.linspace(0, 1, num_points), tck)
    interpolated_points = np.concatenate([xs[:, None], ys[:, None]], axis=-1).astype(int)

    # Remove duplicates
    interpolated_points = filter_duplicates(interpolated_points)

    return interpolated_points


class XRayCatheterDataset(SegmentationDataset):
    LABELS = {
        'ETT': 0,
        'NGT': 1,
        'CVC': 2,
        'Swa': 3,
    }

    @classmethod
    def load_mask(cls, item):
        num_classes = 4
        dx = dy = 10

        mask = np.zeros((item['height'], item['width'], num_classes))
        for label, points in item['mask_points']:
            if len(points) < 2:
                continue

            for x, y in interpolate_mask(points):
                ymin, ymax = max(0, y - dy), min(item['height'], y + dy)
                xmin, xmax = max(0, x - dx), min(item['width'], x + dx)
                mask[ymin:ymax, xmin:xmax, cls.LABELS[label]] = 1

        return mask

    @classmethod
    def load_items(cls, annotations_csv: str, images_dir: str, labels_csv: str) -> list:
        anno_df = load_train_annotations(annotations_csv)
        labels_df = load_train_labels(labels_csv)
        anno_df = anno_df.merge(labels_df.loc[:, ['StudyInstanceUID', 'PatientID', 'fold']], on='StudyInstanceUID')

        def parse_mask(row):
            return row.label[:3], filter_duplicates(np.asarray(literal_eval(row.data)))

        items, not_found = [], 0
        for instance_uid, instance_group in anno_df.groupby('StudyInstanceUID'):
            image_file = osp.join(images_dir, f'{instance_uid}.jpg')
            if not osp.exists(image_file):
                not_found += 1
                continue

            width, height = Image.open(image_file).size
            items.append(
                {
                    'instance_uid': instance_uid,
                    'image_file': image_file,
                    'mask_points': [parse_mask(row) for row in instance_group.itertuples()],
                    'patient_id': instance_group.PatientID.iloc[0],
                    'fold': instance_group.fold.iloc[0],
                    'width': width,
                    'height': height,
                }
            )

        print('Dataset successfully loaded.')
        if not_found > 0:
            n_images = anno_df.StudyInstanceUID.nunique()
            print(f'Not found {not_found} images out of {n_images}. They was skipped.')

        return items

    @classmethod
    def create(cls, annotations_csv: str, images_dir: str, labels_csv: str, transform: A.BasicTransform = None):
        items = cls.load_items(annotations_csv, images_dir, labels_csv)
        return cls(items, classes=[], transform=transform)


def load_items(
    project: str,
    images_dir: str,
    labels_csv: str,
    masks_dir: Optional[str] = None,
    annotations_csv: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if 'lung' in project:
        assert masks_dir is not None
        return XRayLungDataset.load_items(labels_csv, images_dir, masks_dir)

    if 'catheter' in project:
        assert annotations_csv is not None
        return XRayCatheterDataset.load_items(annotations_csv, images_dir, labels_csv)

    raise ValueError(f'Unexpected task: {project}. Project should contain one of: lung, catheter')


if __name__ == '__main__':
    catheter_dataset = XRayCatheterDataset.create(
        annotations_csv='data/train_annotations.csv', images_dir='data/train', labels_csv='data/train.csv'
    )
