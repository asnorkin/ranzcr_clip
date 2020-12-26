import os
import os.path as osp
import warnings
import zipfile
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from classification.datamodule import XRayClassificationDataModule
from classification.loss import batch_roc_auc
from classification.module import XRayClassificationModule

warnings.filterwarnings("ignore", category=UserWarning)


def add_program_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    # General
    parser.add_argument('--project', type=str, default='resnext50_32x4d')
    parser.add_argument('--experiment', type=str, default='train')
    parser.add_argument('--monitor_mode', type=str, default='min')

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

    args.archives_dir = osp.join(args.work_dir, 'archived_checkpoints')
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
        filename='',
        save_top_k=1,
        save_last=True,
        monitor='val_monitor',
        mode=args.monitor_mode)


def tensorboard_logger(args, fold=-1):
    prefix = f'fold{fold}' if fold >= 0 else ''
    version = f'fold{fold}' if fold >= 0 else None
    return TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment,
        prefix=prefix,
        version=version,
        default_hp_metric=False)


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
        logger=logger)

    # Fit
    trainer.fit(model, datamodule=data)

    # Calculate OOF predictions
    if fold >= 0:
        trainer.test(model, test_dataloaders=data.val_dataloader())
        return data.val_indices[fold], model.test_probabilities


def archive_checkpoints(checkpoints_dir, config_file, archive_file):
    create_if_not_exist(osp.dirname(archive_file))

    with zipfile.ZipFile(archive_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(config_file, arcname='config.yml')
        for fname in os.listdir(checkpoints_dir):
            if fname.startswith('fold'):
                zipf.write(osp.join(checkpoints_dir, fname), arcname=fname)


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

    # Verbose
    oof_roc_auc = batch_roc_auc(
        targets=torch.as_tensor([item['target'] for item in data.items]),
        probabilities=oof_probabilities)
    print(f'OOF ROC AUC: {oof_roc_auc:.3f}')

    # Save OOF probabilities
    np.save(osp.join(args.checkpoints_dir, 'oof_probabilities.npy'), oof_probabilities)

    # Save checkpoints
    archive_name = f'{args.experiment}'
    archive_name += f'_{args.lr}lr{args.cv_folds}f{args.num_epochs}e{args.batch_size}b'
    archive_name += f'_oofrocauc{oof_roc_auc}'
    archive_file = osp.join(args.archives_dir, archive_name)
    archive_checkpoints(args.checkpoints_dir, args.config_file, archive_file)


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
    main(config_args())
