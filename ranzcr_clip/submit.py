import os.path as osp
from argparse import ArgumentParser, Namespace

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from classification.dataset import InferenceXRayDataset
from classification.modelzoo import ModelConfig
from predictors import FoldPredictor


TARGET_NAMES = [
    'ETT - Abnormal',
    'ETT - Borderline',
    'ETT - Normal',
    'NGT - Abnormal',
    'NGT - Borderline',
    'NGT - Incompletely Imaged',
    'NGT - Normal',
    'CVC - Abnormal',
    'CVC - Borderline',
    'CVC - Normal',
    'Swan Ganz Catheter Present',
]


def config_args() -> Namespace:
    ap = ArgumentParser()

    ap.add_argument('--images_dir', type=str, required=True)
    ap.add_argument('--checkpoints_dir', type=str, required=True)
    ap.add_argument('--output_dir', type=str, default='.')
    ap.add_argument('--output', type=str, default='mean')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--tta', action='store_true')
    ap.add_argument('--power', type=float, default=1.0)
    ap.add_argument('--folds', type=str, default=None)
    ap.add_argument('--debug', action='store_true')

    args = ap.parse_args()

    return args


def create_batch_generator(args: Namespace, model_config: ModelConfig) -> DataLoader:
    transform = A.Compose(
        [
            A.Resize(height=model_config.input_height, width=model_config.input_width, always_apply=True),
            A.CLAHE(always_apply=True),
            A.Normalize(mean=0.482, std=0.220, always_apply=True),  # Ranzcr
            ToTensorV2(always_apply=True),
        ]
    )

    dataset = InferenceXRayDataset.create(args.images_dir, transform=transform)

    if args.debug:
        dataset.indices = dataset.indices[:50]

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )


def save(predictions: np.ndarray, image_uids: list, output_dir: str) -> pd.DataFrame:
    predictions = pd.DataFrame(data=predictions, columns=TARGET_NAMES)
    predictions['StudyInstanceUID'] = image_uids

    predictions.to_csv(osp.join(output_dir, 'submission.csv'), index=False)
    return predictions


def main(args: Namespace):
    print(f'Submit params:\n{args}')

    # Create batch generator and predictor
    predictor = FoldPredictor.create_from_checkpoints(args.checkpoints_dir, folds=args.folds)
    batch_generator = create_batch_generator(args, model_config=predictor.config)

    # Make predictions
    predictions, image_uids = [], []
    for batch in tqdm(batch_generator, desc='Make predictions', unit='batch'):
        predictions.append(predictor.predict_batch(batch, tta=args.tta, power=args.power, output=args.output))
        image_uids.extend(batch['instance_uid'])

    # Aggregate
    predictions = torch.cat(predictions).to(torch.float32)
    predictions = predictions.cpu().numpy()

    # Save submission
    save(predictions, image_uids, args.output_dir)


if __name__ == '__main__':
    main(config_args())
