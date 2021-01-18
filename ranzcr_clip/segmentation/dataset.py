import os.path as osp

import albumentations as A
import pandas as pd
from tqdm import tqdm

from common.dataset import ImageItemsDataset


class XRayLungDataset(ImageItemsDataset):
    def _load_sample(self, index: int) -> dict:
        index = self._index(index)
        item = self.items[index]

        sample = {
            'image': self.load_image(item['image_file']),
            'mask': self.load_image(item['mask_file']) / 255.0,
            'index': index,
        }

        return sample

    @classmethod
    def load_items(cls, labels_csv: str, images_dir: str, masks_dir: str) -> list:
        labels_df = pd.read_csv(labels_csv)

        # Fix items order
        labels_df.sort_values(by='StudyInstanceUID', inplace=True)

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

        return items

    @classmethod
    def create(cls, labels_csv: str, images_dir: str, masks_dir: str, transform: A.BasicTransform = None):
        items = cls.load_items(labels_csv, images_dir, masks_dir)
        return cls(items, classes=[], transform=transform)
