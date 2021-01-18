import os
import os.path as osp
import zipfile
from argparse import Namespace
from typing import Optional

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from common.fs_utils import create_if_not_exist


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
