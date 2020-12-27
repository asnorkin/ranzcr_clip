import os
import os.path as osp
from argparse import ArgumentParser

import pandas as pd
import torch
from tqdm import tqdm

from classification.dataset import XRayDataset
from predictors import FoldPredictor, TorchModelPredictor


def config_args():
    ap = ArgumentParser()

    ap.add_argument('--images_dir', type=str, required=True)
    ap.add_argument('--checkpoints_dir', type=str, required=True)
    ap.add_argument('--output_dir', type=str, default='.')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--predictor_type', type=str, default='fold', choices=['fold', 'single'])

    args = ap.parse_args()

    return args


class BatchGenerator:
    def __init__(self, dirpath, batch_size):
        self.batch_size = batch_size
        self.dirpath = dirpath
        self.files = [osp.join(dirpath, fname) for fname in os.listdir(dirpath)]
        self.cur_index = 0

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_index >= len(self.files):
            raise StopIteration

        batch_size = min(self.batch_size, len(self.files) - self.cur_index)

        batch, batch_iuids = [], []
        for i in range(batch_size):
            image_file = self.files[self.cur_index]
            batch.append(XRayDataset.load_image(image_file))
            batch_iuids.append(self.iuid(image_file))
            self.cur_index += 1

        return {
            'image': batch,
            'instance_uid': batch_iuids,
        }

    @staticmethod
    def iuid(image_file):
        return osp.splitext(osp.basename(image_file))[0]


def save(predictions, output_dir):
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
    batch_generator = BatchGenerator(args.images_dir, args.batch_size)
    predictor = create_predictor(args)

    # Make predictions
    predictions = []
    for batch in tqdm(batch_generator, desc='Make predictions', unit='batch'):
        predictions.append(predictor.predict_batch(batch))
    predictions = torch.cat(predictions).cpu().numpy()

    # Save submission
    save(predictions, args.output_dir)


if __name__ == '__main__':
    main(config_args())
