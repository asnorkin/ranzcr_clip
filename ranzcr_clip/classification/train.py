import os
import os.path as osp
import warnings
from argparse import ArgumentParser, Namespace
from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from prettytable import PrettyTable
from sklearn.metrics import classification_report
from torch import distributed as dist

from classification.datamodule import XRayClassificationDataModule
from classification.dataset import XRayDataset
from classification.experiment import Experiment
from classification.loss import batch_auc_roc, reduce_auc_roc
from classification.module import XRayClassificationModule

from common.fs_utils import create_dirs
from common.pl_utils import (
    archive_checkpoints,
    checkpoint_callback,
    early_stopping_callback,
    get_checkpoint,
    lr_monitor_callback,
    parse_args,
    tensorboard_logger,
)

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

    return parse_args(parser)


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
    ckpt_callback = checkpoint_callback(args, fold=fold, val_metric='val_roc_auc')
    callbacks.append(ckpt_callback)
    callbacks.append(lr_monitor_callback())

    if args.scheduler == 'reducelronplateau':
        callbacks.append(early_stopping_callback(args))

    # Existing checkpoint
    checkpoint_file = get_checkpoint(args.checkpoints_dir, fold)
    if checkpoint_file is not None:
        if args.exist_checkpoint == 'resume' or args.finetune:
            args.resume_from_checkpoint = checkpoint_file
        elif args.exist_checkpoint == 'remove':
            os.remove(checkpoint_file)

    elif args.finetune:
        raise RuntimeError(
            f'Misconfiguration: --finetune flag is set, but there is no checkpoint file found for fold {fold}'
        )

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
    lung_masks_dir = args.lung_masks_dir if args.lung_masks else None
    items, classes = XRayDataset.load_items(
        labels_csv=args.labels_csv, images_dir=args.images_dir, lung_masks_dir=lung_masks_dir
    )

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
