import os.path as osp
from argparse import ArgumentParser

import albumentations as A
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from classification.dataset import InferenceXRayDataset
from predictors import FoldPredictor, TorchModelPredictor


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


def config_args():
    ap = ArgumentParser()

    ap.add_argument('--images_dir', type=str, required=True)
    ap.add_argument('--checkpoints_dir', type=str, required=True)
    ap.add_argument('--output_dir', type=str, default='.')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--predictor_type', type=str, default='fold', choices=['fold', 'single'])
    ap.add_argument('--tta', action='store_true')

    args = ap.parse_args()

    return args


def create_batch_generator(args, model_config):
    transform = A.Compose(
        [
            A.Resize(height=model_config.input_height, width=model_config.input_width, always_apply=True),
            A.FromFloat('uint8', always_apply=True),
            A.CLAHE(always_apply=True),
            # A.Normalize(mean=0.449, std=0.226, always_apply=True),    # ImageNet
            A.Normalize(mean=0.482, std=0.220, always_apply=True),  # Ranzcr
            ToTensorV2(always_apply=True),
        ]
    )

    dataset = InferenceXRayDataset.create(args.images_dir, transform=transform)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )


def save(predictions, image_uids, output_dir):
    predictions = pd.DataFrame(data=predictions, columns=TARGET_NAMES)
    predictions['StudyInstanceUID'] = image_uids

    predictions.to_csv(osp.join(output_dir, 'submission.csv'), index=False)
    return predictions


def create_predictor(args):
    if args.predictor_type == 'fold':
        predictor = FoldPredictor.create_from_checkpoints(args.checkpoints_dir)
    elif args.predictor_type == 'single':
        predictor = TorchModelPredictor.create_from_checkpoints(args.checkpoints_dir)
    else:
        raise TypeError(f'Unexpected predictor type: {args.predictor_type}')

    return predictor


def main(args):
    # Create batch generator and predictor
    predictor = create_predictor(args)
    batch_generator = create_batch_generator(args, model_config=predictor.config)

    # Make predictions
    predictions, image_uids = [], []
    for batch in tqdm(batch_generator, desc='Make predictions', unit='batch'):
        predictions.append(predictor.predict_batch(batch, tta=args.tta))
        image_uids.extend(batch['instance_uid'])

    # Aggregate
    predictions = torch.cat(predictions).to(torch.float32)
    predictions = predictions.cpu().numpy()

    # Save submission
    save(predictions, image_uids, args.output_dir)


if __name__ == '__main__':
    main(config_args())