from argparse import ArgumentParser, Namespace
from math import ceil

import pytorch_lightning as pl

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from classification import modelzoo
from classification.loss import batch_auc_roc, BCEWithLogitsLoss
from classification.modelzoo import ModelConfig


class XRayClassificationModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        # Config
        self.config = ModelConfig(hparams.config_file)

        # Hyperparameters
        self.hparams = hparams

        # Model
        self.model = self.build_model(self.config)

        # Criterion
        self.criterion = BCEWithLogitsLoss(epsilon=self.hparams.smoothing_epsilon)

        # Placeholders
        self.test_labels = None
        self.test_probabilities = None

    def forward(self, images):
        return self.model(images)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay)

        scheduler = self._configure_scheduler(optimizer)

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
        return self._step(batch, batch_idx, stage='val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage='test')

    def on_validation_epoch_start(self):
        self._on_epoch_start()

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, stage='val')

    def on_test_epoch_start(self):
        self._on_epoch_start()

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, stage='test')

    def _on_epoch_start(self):
        self.test_probabilities = []
        self.test_labels = []

    def _epoch_end(self, _outputs, stage='val'):
        self.test_probabilities = torch.cat(self.test_probabilities)
        self.test_labels = torch.cat(self.test_labels)

        roc_auc = batch_auc_roc(self.test_labels, self.test_probabilities)
        roc_auc = torch.as_tensor(roc_auc)

        if stage == 'val':
            self.log(f'{stage}_roc_auc', roc_auc, logger=False, prog_bar=True, sync_dist=True)
            self.log(f'metrics/{stage}_roc_auc', roc_auc, logger=True, prog_bar=False, sync_dist=True)

        if stage == 'val':
            self.log('val_monitor', -roc_auc, sync_dist=True)

    def _step(self, batch, _batch_idx, stage):
        logits = self.forward(batch['image'])
        losses, metrics = self.loss(logits, batch)

        if stage in {'train', 'val'}:
            self._log(losses, metrics, stage)

        if stage in {'val', 'test'}:
            if self.hparams.use_tta:
                logits = self._tta(batch, logits)

            probabilities = logits.sigmoid()
            self.test_probabilities.append(probabilities)
            self.test_labels.append(batch['target'])

        return losses['total']

    def _tta(self, batch, logits):
        logits_hflip = self.forward(torch.flip(batch['image'], dims=(-1,)))
        return (logits + logits_hflip) / 2

    def _log(self, losses, metrics, stage):
        progress_bar, logs = self._get_progress_bar_and_logs(losses, metrics, stage)

        if len(progress_bar) > 0:
            self.log_dict(progress_bar, prog_bar=True, logger=False, sync_dist=True)

        if len(logs) > 0:
            self.log_dict(logs, prog_bar=False, logger=True, sync_dist=True)

    def _configure_scheduler(self, optimizer):
        if self.hparams.scheduler == 'onecyclelr':
            opt_step_period = self.hparams.batch_size * self.trainer.accumulate_grad_batches
            steps_per_epoch = ceil(len(self.trainer.datamodule.train_dataset) / opt_step_period)
            scheduler = {
                'scheduler': OneCycleLR(
                    optimizer, max_lr=self.hparams.lr, pct_start=self.hparams.lr_pct_start,
                    div_factor=self.hparams.lr_div_factor, steps_per_epoch=steps_per_epoch,
                    epochs=self.hparams.max_epochs),
                'interval': 'step',
            }
        elif self.hparams.scheduler == 'reducelronplateau':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=self.hparams.lr_factor, patience=self.hparams.lr_patience,
                    mode=self.hparams.monitor_mode, verbose=True),
                'monitor': 'val_monitor',
                'interval': 'epoch',
                'frequency': self.hparams.check_val_every_n_epoch,
            }
        else:
            raise ValueError(f'Unexpected scheduler type: {self.hparams.scheduler}')

        return scheduler

    @staticmethod
    def _get_progress_bar_and_logs(losses, metrics, stage):
        # Progress bar
        progress_bar = {}
        if stage != 'train':
            progress_bar[f'{stage}_loss'] = losses['total']

        # Logs
        logs = dict()
        if len(losses) > 0:
            logs.update({f'losses/{stage}_{loss_name}': loss_value
                         for loss_name, loss_value in losses.items()})
        if len(metrics) > 0:
            logs.update({f'metrics/{stage}_{metric_name}': metric_value
                         for metric_name, metric_value in metrics.items()})

        return progress_bar, logs

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Loss
        parser.add_argument('--smoothing_epsilon', type=float, default=0.0)

        # Optimizer and scheduler
        parser.add_argument('--weight_decay', type=float, default=1e-6)
        parser.add_argument('--scheduler', type=str, default='reducelronplateau',
                            choices=['reducelronplateau', 'onecyclelr'])

        # OneCycleLR
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--lr_pct_start', type=float, default=0.1)
        parser.add_argument('--lr_div_factor', type=float, default=1000.)
        parser.add_argument('--lr_final_div_factor', type=float, default=500.)

        # ReduceLROnPlateau
        parser.add_argument('--lr_factor', type=float, default=0.1)
        parser.add_argument('--lr_patience', type=int, default=1)

        # Early stopping
        parser.add_argument('--es_patience', type=int, default=3)

        # TTA
        parser.add_argument('--use_tta', action='store_true')

        return parser

    @staticmethod
    def build_model(config, checkpoint_file=None):
        model_builder = getattr(modelzoo, config.model_type, None)
        if model_builder is None:
            raise ValueError(f'Unexpected model_type: {config.model_type}')

        model = model_builder(num_classes=config.num_classes, pretrained=checkpoint_file is None)

        if checkpoint_file is not None:
            ckpt = torch.load(checkpoint_file, map_location='cpu')
            state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
            model.load_state_dict(state_dict, strict=True)

        return model
