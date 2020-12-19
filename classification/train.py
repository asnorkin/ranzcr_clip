import os
import os.path as osp
import warnings
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from classification.datamodule import XRayClassificationDataModule
from classification.module import XRayClassificationModule

warnings.filterwarnings("ignore", category=UserWarning)


def add_program_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    # General
    parser.add_argument('--project', type=str, default='resnext50_32x4d')
    parser.add_argument('--experiment', type=str, default='train')

    # Paths
    parser.add_argument('--work_dir', type=str, default='classification')

    # Seed
    parser.add_argument('--seed', type=int, default=42)

    return parser


def config_args():
    parser = ArgumentParser()

    parser = add_program_specific_args(parser)
    parser = XRayClassificationDataModule.add_data_specific_args(parser)
    parser = XRayClassificationModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    args.config_file = osp.join(args.work_dir, 'models', args.project + '.yml')
    args.checkpoints_dir = f'{args.work_dir}/checkpoints/{args.experiment}'
    args.log_dir = f'{args.work_dir}/logs'

    if args.num_epochs is not None:
        args.max_epochs = args.num_epochs

    return args


def create_if_not_exist(dirpath):
    if not osp.exists(dirpath):
        os.makedirs(dirpath)


def create_dirs(args):
    create_if_not_exist(args.checkpoints_dir)
    create_if_not_exist(args.log_dir)


def checkpoint_callback(args, fold=-1):
    if not osp.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    prefix = f'fold{fold}' if fold >= 0 else ''

    return ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        prefix=prefix,
        save_top_k=1,
        save_last=True,
        monitor='val_monitor',
        mode='min',
    )


def tensorboard_logger(args, fold=-1):
    prefix = f'fold{fold}' if fold >= 0 else ''
    version = f'fold{fold}' if fold >= 0 else None
    return TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment,
        prefix=prefix,
        version=version,
        default_hp_metric=False,
    )


def train_model(args, fold=-1, data=None):
    # Set up seed
    pl.seed_everything(seed=args.seed + fold)

    # If no data provided create it
    if data is None:
        data = XRayClassificationDataModule(args)

    # Setup data fold
    if fold >= 0:
        data.setup_fold(fold)

    # Create model
    model = XRayClassificationModule(args)

    # Create trainer
    logger = tensorboard_logger(args, fold=fold)
    ckpt_callback = checkpoint_callback(args, fold=fold)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[ckpt_callback],
        logger=logger,
    )

    # Fit
    trainer.fit(model, datamodule=data)

    # Calculate OOF predictions
    if fold >= 0:
        trainer.test(model, test_dataloaders=data.val_dataloader())
        return data.val_indices[fold], model.test_probabilities


def cross_validate(args):
    # Create and setup datamodule
    data = XRayClassificationDataModule(args)
    data.setup()

    # OOF probabilities placeholder
    oof_probabilities = np.zeros((len(data.items), 11))

    # Folds loop
    for fold in range(args.cv_folds):
        print(f'FOLD {fold}')
        fold_oof_indices, fold_oof_probabilities = train_model(args, fold=fold, data=data)
        oof_probabilities[fold_oof_indices] = fold_oof_probabilities

    # Save OOF probabilities
    np.save(osp.join(args.checkpoints_dir, 'oof_probabilities.npy'), oof_probabilities)


def main(args):
    # Create checkpoints and logs dirs
    create_dirs(args)

    # Train one model
    if args.cv_folds is None:
        train_model(args)

    # Train fold models
    else:
        cross_validate(args)


if __name__ == '__main__':
    args = config_args()
    main(args)
