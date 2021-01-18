import os
import os.path as osp
from argparse import ArgumentParser, Namespace
from typing import List, Optional

import pytorch_lightning as pl
import torch

from common.fs_utils import create_dirs
from common.pl_utils import (
    checkpoint_callback,
    early_stopping_callback,
    get_checkpoint,
    lr_monitor_callback,
    tensorboard_logger,
)

from segmentation.datamodule import LungSegmentationDataModule
from segmentation.dataset import XRayLungDataset
from segmentation.module import LungSegmentationModule


def add_program_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    # General
    parser.add_argument('--project', type=str, default='unet64_512x512')
    parser.add_argument('--experiment', type=str, default='train')
    parser.add_argument('--monitor_mode', type=str, default='min')
    parser.add_argument('--exist_checkpoint', type=str, default='test', choices=['resume', 'test', 'remove'])
    parser.add_argument('--folds', type=str, default=None)

    # Paths
    parser.add_argument('--work_dir', type=str, default='segmentation')

    # Seed
    parser.add_argument('--seed', type=int, default=42)

    return parser


def config_args() -> Namespace:
    parser = ArgumentParser()

    parser = add_program_specific_args(parser)
    parser = LungSegmentationDataModule.add_data_specific_args(parser)
    parser = LungSegmentationModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    args.config_file = osp.join(args.work_dir, 'models', args.project + '.yml')
    args.checkpoints_dir = f'{args.work_dir}/checkpoints/{args.experiment}'
    args.log_dir = f'{args.work_dir}/logs'

    if args.num_epochs is not None:
        args.max_epochs = args.num_epochs

    if args.seed is not None:
        args.benchmark = False
        args.deterministic = True

    return args


def train_fold(args: Namespace, fold: int = -1, items: Optional[List] = None) -> Optional[pl.Trainer]:
    pl.seed_everything(seed=args.seed + fold)

    # Set up fold
    args.fold = fold

    # Create and setup data
    data = LungSegmentationDataModule(args, items=items)
    data.setup()

    # Create model
    model = LungSegmentationModule(args)

    # Create logger
    logger = tensorboard_logger(args, fold=fold)

    # Create callbacks
    callbacks = []
    ckpt_callback = checkpoint_callback(args, fold=fold)
    callbacks.append(ckpt_callback)
    callbacks.append(lr_monitor_callback())

    if args.scheduler == 'reducelronplateau':
        callbacks.append(early_stopping_callback(args))

    # Existing checkpoint
    checkpoint_file = get_checkpoint(args.checkpoints_dir, fold)
    if checkpoint_file is not None:
        if args.exist_checkpoint == 'resume':
            args.resume_from_checkpoint = checkpoint_file
        elif args.exist_checkpoint == 'remove':
            os.remove(checkpoint_file)

    # Create trainer and fit
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule=data)

    # Load best weights and test
    if checkpoint_file is not None and args.exist_checkpoint == 'test':
        # Test only
        test_model_path = checkpoint_file

    else:
        # Fit
        trainer.fit(model, datamodule=data)
        test_model_path = ckpt_callback.best_model_path

    # Load best weights for test
    model.load_state_dict(torch.load(test_model_path)['state_dict'])
    model.load_state_dict(torch.load(ckpt_callback.best_model_path)['state_dict'])
    trainer.test(model, test_dataloaders=data.val_dataloader())

    if trainer.global_rank != 0:
        return None

    return trainer


def train(args: Namespace):
    # Load items only once
    items = XRayLungDataset.load_items(
        labels_csv=args.labels_csv, images_dir=args.images_dir, masks_dir=args.masks_dir
    )

    # Folds
    folds = args.folds or sorted({item['fold'] for item in items})

    # Folds loop
    for fold in folds:
        print(f'FOLD {fold}')
        train_fold(args, fold=fold, items=items)


def main(args: Namespace) -> None:
    # Create checkpoints and logs dirs
    create_dirs(args)

    # Train
    train(args)


if __name__ == '__main__':
    main(config_args())
