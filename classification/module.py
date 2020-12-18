from argparse import ArgumentParser, Namespace
from math import ceil

import pytorch_lightning as pl

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

from classification.loss import batch_roc_auc, BCEWithLogitsLoss
from classification.model import ModelConfig, resnext50_32x4d


class XRayClassificationModule(pl.LightningModule):
    def __init__(self, hparams, num_classes=11):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        # Config
        self.config = ModelConfig(hparams.config_file)

        # Hyperparameters
        self.hparams = hparams

        # Model
        self.model = resnext50_32x4d(num_classes=num_classes)

        # Criterion
        self.criterion = BCEWithLogitsLoss(epsilon=self.hparams.smoothing_epsilon)

        # Placeholders
        self.test_labels = None
        self.test_probabilities = None

    def forward(self, images):
        return self.model(images)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), lr=self.hparams.lr,  weight_decay=self.hparams.weight_decay)

        opt_step_period = self.hparams.batch_size * self.trainer.accumulate_grad_batches
        steps_per_epoch = ceil(len(self.trainer.datamodule.train_dataset) / opt_step_period)
        scheduler = {
            'scheduler': OneCycleLR(
                optimizer, max_lr=self.hparams.lr, pct_start=self.hparams.lr_pct_start,
                div_factor=self.hparams.lr_div_factor, steps_per_epoch=steps_per_epoch,
                epochs=self.hparams.max_epochs),
            'interval': 'step',
        }

        return [optimizer], [scheduler]

    def loss(self, logits, batch):
        losses = dict()
        losses['total'] = self.criterion(logits, batch['target'])

        # predictions = torch.argmax(logits, dim=-1)

        metrics = dict()
        # metrics['acc'] = (batch['target'] == predictions).float().mean()

        return losses, metrics

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._step(batch, batch_idx, stage='val')

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._step(batch, batch_idx, stage='test')

    def on_validation_epoch_start(self):
        self._on_epoch_start()

    def on_validation_epoch_end(self):
        self._on_epoch_end(stage='val')

    def on_test_epoch_start(self):
        self._on_epoch_start()

    def on_test_epoch_end(self):
        self._on_epoch_end(stage='test')

    def _on_epoch_start(self):
        self.test_probabilities = []
        self.test_labels = []

    def _on_epoch_end(self, stage='val'):
        self.test_probabilities = torch.cat(self.test_probabilities).cpu().numpy()
        self.test_labels = torch.cat(self.test_labels).cpu().numpy()

        roc_auc = batch_roc_auc(self.test_labels, self.test_probabilities)
        roc_auc = torch.as_tensor(roc_auc)

        self.log('val_roc_auc', roc_auc, logger=False, prog_bar=True)
        self.log(f'metrics/{stage}_{roc_auc}', roc_auc, logger=True, prog_bar=False)

        if stage == 'val':
            self.log('val_monitor', -roc_auc)

    def _step(self, batch, batch_idx, stage):
        logits = self.forward(batch['image'])
        losses, metrics = self.loss(logits, batch)
        self._log(losses, metrics, stage)

        if stage in {'val', 'test'}:
            probabilities = logits.sigmoid()
            self.test_probabilities.append(probabilities)
            self.test_labels.append(batch['target'])

        return losses['total']

    def _log(self, losses, metrics, stage):
        progress_bar, logs = self._get_progress_bar_and_logs(losses, metrics, stage)

        if len(progress_bar) > 0:
            self.log_dict(progress_bar, prog_bar=True, logger=False)

        if len(logs) > 0:
            self.log_dict(logs, prog_bar=False, logger=True)

    def _get_progress_bar_and_logs(self, losses, metrics, stage):
        # Progress bar
        prefix = '' if stage == 'train' else f'{stage}_'
        progress_bar = {}
        if stage != 'train':
            progress_bar[f'{prefix}loss'] = losses['total']

        # Logs
        logs = dict()
        logs.update({f'losses/{stage}_{loss_name}': loss_value for loss_name, loss_value in losses.items()})
        logs.update({f'metrics/{stage}_{metric_name}': metric_value for metric_name, metric_value in metrics.items()})

        return progress_bar, logs

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Loss
        parser.add_argument('--smoothing_epsilon', type=float, default=0.0)

        # Optimizer
        parser.add_argument('--weight_decay', type=float, default=1e-6)

        # Learning rate
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--lr_pct_start', type=float, default=0.2)
        parser.add_argument('--lr_div_factor', type=float, default=1e3)

        return parser
