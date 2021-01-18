from argparse import ArgumentParser, Namespace
from math import sqrt
from typing import Union

import pytorch_lightning as pl
from torch.optim import AdamW, Optimizer, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

from common.model_utils import ModelConfig
from segmentation.loss import SegmentationLoss
from segmentation.modelzoo import unet


class LungSegmentationModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        # Config
        self.config = ModelConfig(hparams.config_file)

        # Hyperparameters
        self.hparams = hparams

        # Model
        self.model = self._build_model()

        # Criterion
        self.criterion = SegmentationLoss(
            n_classes=self.config.n_classes,
            dice_weight=self.hparams.dice_weight,
            dice_eps=self.hparams.dice_eps,
        )

    def forward(self, inputs):
        return self.model.forward(inputs)

    def configure_optimizers(self):
        optimizer = self._configure_optimizer()
        scheduler = self._configure_scheduler(optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        return self._step(batch, batch_idx, stage='train')

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        return self._step(batch, batch_idx, stage='val')

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        return self._step(batch, batch_idx, stage='test')

    def _step(self, batch: dict, _batch_idx: int, stage: str) -> dict:
        logits = self.forward(batch['image'])[:, 0]
        losses = self.criterion(logits, batch['mask'])

        self.log('dice', losses['dice'], prog_bar=True, logger=True)

        if stage == 'val':
            self.log_dict(
                {
                    'val_loss': losses['total'],
                    'val_dice': losses['dice'],
                },
                prog_bar=True,
                logger=True,
            )
            self.log('val_monitor', -losses['dice'])

        return losses['total']

    def _configure_optimizer(self) -> Union[AdamW, RMSprop]:
        if self.hparams.optimizer == 'rmsprop':
            optimizer = RMSprop(
                self.model.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                momentum=self.hparams.momentum,
            )

        elif self.hparams.optimizer == 'adamw':
            optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        else:
            raise ValueError(f'Unexpected optimizer type: {self.hparams.optimizer}')

        return optimizer

    def _configure_scheduler(self, optimizer: Optimizer) -> dict:
        if self.hparams.scheduler == 'reducelronplateau':
            scheduler = {
                'scheduler': ReduceLROnPlateau(
                    optimizer,
                    factor=self.hparams.lr_factor,
                    patience=self.hparams.lr_patience,
                    mode=self.hparams.monitor_mode,
                    threshold=1e-4,
                    threshold_mode='abs',
                    verbose=True,
                ),
                'monitor': 'val_monitor',
                'interval': 'epoch',
                'frequency': self.hparams.check_val_every_n_epoch,
            }
        else:
            raise ValueError(f'Unexpected scheduler type: {self.hparams.scheduler}')

        return scheduler

    def _build_model(self):
        return unet(self.config)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Loss
        parser.add_argument('--dice_eps', type=float, default=1e-3)
        parser.add_argument('--dice_weight', type=float, default=1.0)

        # Optimizer
        parser.add_argument('--optimizer', type=str, choices=['rmsprop', 'adamw'], default='rmsprop')
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-8)
        parser.add_argument('--momentum', type=float, default=0.9)

        # Scheduler
        parser.add_argument('--scheduler', type=str, choices=['reducelronplateau'], default='reducelronplateau')
        parser.add_argument('--lr_factor', type=float, default=sqrt(0.1))
        parser.add_argument('--lr_patience', type=int, default=0)

        # Early stopping
        parser.add_argument('--es_patience', type=int, default=3)

        return parser
