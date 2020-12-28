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


def config_args():
    ap = ArgumentParser()

    ap.add_argument('--images_dir', type=str, required=True)
    ap.add_argument('--checkpoints_dir', type=str, required=True)
    ap.add_argument('--output_dir', type=str, default='.')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--predictor_type', type=str, default='fold', choices=['fold', 'single'])

    args = ap.parse_args()

    return args


def create_batch_generator(args, model_config):
    transform = A.Compose([
        A.Resize(height=model_config.input_height, width=model_config.input_width),
        A.Normalize(max_pixel_value=1.0),
        ToTensorV2(),
    ])

    dataset = InferenceXRayDataset.create(args.images_dir, cache_images=True, transform=transform)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False)


def save(predictions, image_uids, output_dir):
    columns = [
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
        'Swan Ganz Catheter Present']

    predictions = pd.DataFrame(data=predictions, columns=columns)
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
        predictions.append(predictor.predict_batch(batch, preprocess=False))
        image_uids.extend(batch['instance_uid'])
    predictions = torch.cat(predictions).cpu().numpy().astype(int)

    # Save submission
    save(predictions, image_uids, args.output_dir)


if __name__ == '__main__':
    main(config_args())
