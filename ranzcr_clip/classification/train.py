import os
import os.path as osp
import warnings
import zipfile
from argparse import ArgumentParser, Namespace
from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from prettytable import PrettyTable
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report
from torch import distributed as dist

from classification.datamodule import XRayClassificationDataModule
from classification.dataset import XRayDataset
from classification.experiment import Experiment
from classification.loss import batch_auc_roc, reduce_auc_roc
from classification.module import XRayClassificationModule
from submit import TARGET_NAMES

warnings.filterwarnings("ignore", category=UserWarning)


def add_program_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    # General
    parser.add_argument('--project', type=str, default='efficientnet_b0')
    parser.add_argument('--experiment', type=str, default='train')
    parser.add_argument('--monitor_mode', type=str, default='min')
    parser.add_argument('--exist_checkpoint', type=str, default='test', choices=['resume', 'test', 'remove'])
    parser.add_argument('--folds', type=str, default=None)

    # Paths
    parser.add_argument('--work_dir', type=str, default='classification')

    # Seed
    parser.add_argument('--seed', type=int, default=42)

    return parser


def config_args() -> Namespace:
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

    if args.seed is not None:
        args.benchmark = False
        args.deterministic = True

    if args.folds is not None:
        args.folds = list(map(int, args.folds.split(',')))

    return args


def create_if_not_exist(dirpath: str):
    if not osp.exists(dirpath):
        os.makedirs(dirpath)


def create_dirs(args: Namespace):
    create_if_not_exist(args.checkpoints_dir)
    create_if_not_exist(args.log_dir)


def checkpoint_callback(args: Namespace, fold: int = -1) -> ModelCheckpoint:
    if not osp.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    filename = f'fold{fold}' if fold >= 0 else 'single'
    filename += '-{epoch:02d}-{val_roc_auc:.3f}'

    return ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        filename=filename,
        save_top_k=1,
        save_last=False,
        monitor='val_monitor',
        mode=args.monitor_mode,
    )


def early_stopping_callback(args: Namespace) -> EarlyStopping:
    return EarlyStopping(monitor='val_monitor', mode=args.monitor_mode, patience=args.es_patience, verbose=True)


def tensorboard_logger(args: Namespace, fold: int = -1) -> TensorBoardLogger:
    version = f'fold{fold}' if fold >= 0 else 'single'
    return TensorBoardLogger(save_dir=args.log_dir, name=args.experiment, version=version, default_hp_metric=False)


def lr_monitor_callback() -> LearningRateMonitor:
    return LearningRateMonitor(log_momentum=False)


def archive_checkpoints(args: Namespace, oof_roc_auc: float, folds: list):
    print('Archive checkpoints..')

    # Archive file
    archive_name = f'{args.experiment}_f{"".join(map(str, folds))}_auc{oof_roc_auc:.3f}'
    archive_file = osp.join(args.archives_dir, archive_name + '.zip')

    # Create archived checkpoints dir
    create_if_not_exist(osp.dirname(archive_file))

    # Get checkpoints files
    checkpoints_files = [fname for fname in os.listdir(args.checkpoints_dir) if fname.startswith('fold')]

    # Archive
    with zipfile.ZipFile(archive_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(args.config_file, arcname='config.yml')
        for fname in checkpoints_files:
            arcname = fname.replace('=', '')  # Remove kaggle illegal character '='
            zipf.write(osp.join(args.checkpoints_dir, fname), arcname=arcname)

    print('Checkpoints successfully archived!')


def get_checkpoint(checkpoints_dir: str, fold: int) -> Optional[str]:
    checkpoint_files = [
        osp.join(checkpoints_dir, fname) for fname in os.listdir(checkpoints_dir) if fname.startswith(f'fold{fold}-')
    ]

    if len(checkpoint_files) > 1:
        msg = '\n\t' + '\n\t'.join(checkpoint_files)
        print(f'Found many checkpoint files:{msg}')

    if len(checkpoint_files) > 0:
        checkpoint_file = checkpoint_files[0]
        print(f'Found fold{fold} checkpoint: {checkpoint_file}')
        return checkpoint_file

    return None


def report(probabilities: np.ndarray, labels: np.ndarray, checkpoints_dir: Optional[str] = None) -> float:
    # Use only calculated already values
    probabilities = probabilities[~np.all(probabilities == -1, axis=1)]
    labels = labels[~np.all(labels == -1, axis=1)]

    # OOF ROC AUC
    oof_roc_auc_values = batch_auc_roc(
        targets=torch.from_numpy(labels), probabilities=torch.from_numpy(probabilities), reduction=None
    )
    oof_roc_auc = reduce_auc_roc(oof_roc_auc_values, reduction='mean').item()
    print(f'OOF ROC AUC: {oof_roc_auc:.3f}')

    num_targets = labels.sum(axis=0).astype(int)

    # Create sklearn classification report
    predictions = np.where(probabilities > 0.5, 1, 0)
    _report = classification_report(predictions, labels, target_names=TARGET_NAMES, output_dict=True)

    # Add ROC AUC to the report
    for i, target in enumerate(TARGET_NAMES):
        _report[target]['roc_auc'] = oof_roc_auc_values[i].item()

    # Covert report to the matrix
    rows = []
    for i, target in enumerate(TARGET_NAMES):
        rows.append(
            [
                target,
                _report[target]['precision'],
                _report[target]['recall'],
                _report[target]['f1-score'],
                _report[target]['roc_auc'],
                _report[target]['support'],
                num_targets[i].item(),
            ]
        )

    # Add macro average
    def _macro(key: str):
        return np.mean([_report[target_name][key] for target_name in TARGET_NAMES])

    rows.append(
        [
            'Macro total',
            _macro('precision'),
            _macro('recall'),
            _macro('f1-score'),
            oof_roc_auc,
            len(predictions),
            len(labels),
        ]
    )

    # Generate table
    columns = ['Class', 'Precision', 'Recall', 'F1-score', 'ROC AUC', 'Predicted objects', 'GT objects']

    report_table = PrettyTable(field_names=columns, float_format='.3')
    report_table.add_rows(rows)
    print(report_table)

    # Save report to csv
    if checkpoints_dir is not None:
        pd.DataFrame(data=rows, columns=columns).to_csv(osp.join(checkpoints_dir, 'report.csv'), index=False)

    return oof_roc_auc


def train_fold(
    args: Namespace, fold: int = -1, items: Optional[List] = None, classes: Optional[List] = None
) -> Optional[pl.Trainer]:
    # Set up seed
    pl.seed_everything(seed=args.seed + fold)

    # Set up fold
    args.fold = fold

    # Create and setup data
    data = XRayClassificationDataModule(args, items=items, classes=classes)
    data.setup()

    # Create model
    model = XRayClassificationModule(args)
    model.setup(targets=data.train_dataset.targets)

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

    # Create trainer
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)

    if checkpoint_file is not None and args.exist_checkpoint == 'test':
        # Test only
        test_model_path = checkpoint_file

    else:
        # Fit
        trainer.fit(model, datamodule=data)
        test_model_path = ckpt_callback.best_model_path

    # Load best weights for test
    model.load_state_dict(torch.load(test_model_path)['state_dict'])

    # Calculate OOF predictions
    trainer.test(model, test_dataloaders=data.val_dataloader())

    if trainer.global_rank != 0:
        return None

    return trainer


def train(args: Namespace):
    NUM_FOLDS = 5

    # Load items only once
    items, classes = XRayDataset.load_items(labels_csv=args.labels_csv, images_dir=args.images_dir)

    # Folds
    folds = args.folds or sorted({item['fold'] for item in items})

    # OOF placeholders
    experiment = Experiment(args, num_items=len(items), num_classes=len(classes), num_folds=NUM_FOLDS)

    # Folds loop
    for fold in folds:
        print(f'FOLD {fold}')

        # Train fold model
        fold_trainer = train_fold(args, fold=fold, items=items, classes=classes)
        if fold_trainer is not None:
            experiment.update_state(fold_trainer, fold)

    # Save and verbose only in zero process for DDP
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    # Verbose
    oof_roc_auc = report(
        probabilities=experiment.probabilities, labels=experiment.labels, checkpoints_dir=args.checkpoints_dir
    )

    # Save state
    experiment.save_state()

    # Save checkpoints
    archive_checkpoints(args, oof_roc_auc, folds=folds)


def main(args: Namespace):
    # Create checkpoints and logs dirs
    create_dirs(args)

    # Train
    train(args)


if __name__ == '__main__':
    main(config_args())
