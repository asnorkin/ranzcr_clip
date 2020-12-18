import os
import os.path as osp
import warnings
from argparse import ArgumentParser

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

    return args


def checkpoint_callback(args):
    if not osp.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    return ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        save_top_k=1,
        save_last=True,
        monitor='val_monitor',
        mode='min',
    )


def tensorboard_logger(args):
    return TensorBoardLogger(
        save_dir=osp.join(args.log_dir, 'tensorboard'),
        name=args.experiment,
    )


def main(args):
    # Set up seed
    pl.seed_everything(seed=args.seed)

    # Create data and model modules
    data = XRayClassificationDataModule(args)
    model = XRayClassificationModule(args)

    # Create trainer
    logger = tensorboard_logger(args)
    ckpt_callback = checkpoint_callback(args)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[ckpt_callback],
        logger=logger,
    )

    # Fit
    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    args = config_args()
    main(args)
