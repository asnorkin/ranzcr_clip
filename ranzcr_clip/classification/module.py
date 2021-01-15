from argparse import ArgumentParser, Namespace
from math import ceil, sqrt
from typing import Callable, Optional, Tuple, Union

import pytorch_lightning as pl

import torch
from torch import distributed as dist
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau

from classification import modelzoo
from classification.loss import batch_auc_roc, BCEWithLogitsLoss
from classification.modelzoo import ModelConfig


class XRayClassificationModule(pl.LightningModule):
    def __init__(self, hparams: Namespace):
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
        self.test_indices = None
        self.test_labels = None
        self.test_probabilities = None
        self.test_roc_auc = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = self._configure_scheduler(optimizer)
        return [optimizer], [scheduler]

    def loss(self, logits: torch.Tensor, batch: dict) -> tuple:
        losses = dict()
        losses['total'] = self.criterion(logits, batch['target'])

        # predictions = torch.argmax(logits, dim=-1)

        metrics = dict()
        # metrics['acc'] = (batch['target'] == predictions).float().mean()

        return losses, metrics

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        # warm up lr
        if epoch < self.hparams.lr_warmup_epochs:
            warmup_steps_per_epoch = self.trainer.num_training_batches // self.hparams.accumulate_grad_batches
            warmup_steps = max(1, warmup_steps_per_epoch * self.hparams.lr_warmup_epochs)
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / warmup_steps)
            step_lr = lr_scale * self.hparams.lr
            for pg in optimizer.param_groups:
                pg['lr'] = step_lr

        # update params
        super().optimizer_step(
            epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs
        )

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        return self._step(batch, batch_idx, stage='train')

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        return self._step(batch, batch_idx, stage='val')

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        return self._step(batch, batch_idx, stage='test')

    def on_train_epoch_start(self) -> None:
        if self.hparams.schedule_input_size:
            self._schedule_input_size()

    def on_validation_epoch_start(self):
        self._on_epoch_start()

    def on_test_epoch_start(self) -> None:
        self._on_epoch_start()

    def validation_epoch_end(self, outputs: list) -> None:
        self._epoch_end(outputs, stage='val')

    def test_epoch_end(self, outputs: list) -> None:
        self._epoch_end(outputs, stage='test')

    def setup(self, _stage: Optional[str] = None, targets: Optional[list] = None) -> None:
        if targets is not None and self.hparams.bce_weights:
            # Calculate loss weights
            weights = self.criterion.calculate_weights(targets, alpha=self.hparams.weights_alpha)
            self.criterion.weights = weights.to(self.device).to(self.dtype)

    def _on_epoch_start(self) -> None:
        self.test_indices = []
        self.test_labels = []
        self.test_probabilities = []

    def _epoch_end(self, outputs: list, stage: str = 'val') -> None:
        def _gather(key):
            node_values = torch.cat([output[key] for output in outputs])
            if self.trainer.world_size == 1:
                return node_values

            all_values = [torch.zeros_like(node_values) for _ in range(self.trainer.world_size)]
            dist.barrier()
            dist.all_gather(all_values, node_values)
            all_values = torch.cat(all_values)
            return all_values

        probabilities = _gather('probabilities')
        labels = _gather('labels')

        roc_auc = batch_auc_roc(labels, probabilities)

        if stage in {'val', 'test'}:
            self.log(f'metrics/{stage}_roc_auc', roc_auc, logger=True, prog_bar=False, sync_dist=True)

        if stage == 'val':
            self.log(f'{stage}_roc_auc', roc_auc, logger=False, prog_bar=True, sync_dist=True)
            self.log('val_monitor', -roc_auc, sync_dist=True)

        self.test_indices = _gather('indices')
        self.test_labels = labels
        self.test_probabilities = probabilities
        self.test_roc_auc = roc_auc

    def _step(self, batch: dict, _batch_idx: int, stage: str) -> dict:
        logits = self.forward(batch['image'])
        losses, metrics = self.loss(logits, batch)

        if stage in {'train', 'val'}:
            self._log(losses, metrics, stage)

        if stage in {'val', 'test'}:
            if self.hparams.use_tta:
                logits = self._tta(batch, logits)

            probabilities = logits.sigmoid()
            return {
                'probabilities': probabilities,
                'labels': batch['target'],
                'indices': batch['index'],
            }

        return losses['total']

    def _tta(self, batch: dict, logits: torch.Tensor) -> torch.Tensor:
        logits_hflip = self.forward(torch.flip(batch['image'], dims=(-1,)))
        return (logits + logits_hflip) / 2

    def _log(self, losses: dict, metrics: dict, stage: str) -> None:
        progress_bar, logs = self._get_progress_bar_and_logs(losses, metrics, stage)

        if len(progress_bar) > 0:
            self.log_dict(progress_bar, prog_bar=True, logger=False)

        if len(logs) > 0:
            self.log_dict(logs, prog_bar=False, logger=True)

    def _configure_scheduler(self, optimizer: torch.optim.Optimizer) -> Union[OneCycleLR, ReduceLROnPlateau]:
        if self.hparams.scheduler == 'onecyclelr':
            dataset_size = len(self.trainer.datamodule.train_dataset)
            step_period = self.hparams.batch_size * self.trainer.accumulate_grad_batches * self.trainer.world_size
            steps_per_epoch = ceil(dataset_size / step_period)
            scheduler = {
                'scheduler': OneCycleLR(
                    optimizer,
                    max_lr=self.hparams.lr,
                    pct_start=self.hparams.lr_pct_start,
                    div_factor=self.hparams.lr_div_factor,
                    steps_per_epoch=steps_per_epoch,
                    epochs=self.hparams.max_epochs,
                ),
                'interval': 'step',
            }
        elif self.hparams.scheduler == 'reducelronplateau':
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
        elif self.hparams.scheduler == 'cosineannealingwarmrestarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.hparams.t0,
                T_mult=self.hparams.tmult,
                eta_min=self.hparams.eta_min,
            )
        else:
            raise ValueError(f'Unexpected scheduler type: {self.hparams.scheduler}')

        return scheduler

    def _schedule_input_size(self):
        min_input_size = 256
        max_input_size = 512
        step = 128

        cur_input_size = self.trainer.datamodule.input_size
        scheduler = self.trainer.lr_schedulers[0]['scheduler']
        if self.current_epoch == 0:
            self.trainer.datamodule.setup_input_size(min_input_size)
        elif scheduler.num_bad_epochs > 0 and cur_input_size < max_input_size:
            self.trainer.datamodule.setup_input_size(cur_input_size + step)

    @staticmethod
    def _get_progress_bar_and_logs(losses: dict, metrics: dict, stage: str) -> Tuple[dict, dict]:
        # Progress bar
        progress_bar = {}
        if stage != 'train':
            progress_bar[f'{stage}_loss'] = losses['total']

        # Logs
        logs = dict()
        if len(losses) > 0:
            logs.update({f'losses/{stage}_{loss_name}': loss_value for loss_name, loss_value in losses.items()})
        if len(metrics) > 0:
            logs.update(
                {f'metrics/{stage}_{metric_name}': metric_value for metric_name, metric_value in metrics.items()}
            )

        return progress_bar, logs

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Loss
        parser.add_argument('--smoothing_epsilon', type=float, default=0.0)
        parser.add_argument('--weights_alpha', type=float, default=None, help='None is a mean value')
        parser.add_argument('--bce_weights', action='store_true')

        # Optimizer and scheduler
        parser.add_argument('--weight_decay', type=float, default=1e-6)
        parser.add_argument(
            '--scheduler',
            type=str,
            default='reducelronplateau',
            choices=['cosineannealingwarmrestarts', 'onecyclelr', 'reducelronplateau'],
        )

        # OneCycleLR
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--lr_pct_start', type=float, default=0.05)
        parser.add_argument('--lr_div_factor', type=float, default=1000.0)
        parser.add_argument('--lr_final_div_factor', type=float, default=1000.0)

        # ReduceLROnPlateau
        parser.add_argument('--lr_factor', type=float, default=sqrt(0.1))
        parser.add_argument('--lr_patience', type=int, default=0)

        # CosineAnnealingWarmRestart
        parser.add_argument('--t0', type=int, default=10)
        parser.add_argument('--tmult', type=int, default=1)
        parser.add_argument('--eta_min', type=float, default=1e-6)

        # Warmup
        parser.add_argument('--lr_warmup_epochs', type=int, default=1)

        # Early stopping
        parser.add_argument('--es_patience', type=int, default=3)

        # Other flags
        parser.add_argument('--use_tta', action='store_true')
        parser.add_argument('--schedule_input_size', action='store_true')

        return parser

    @staticmethod
    def build_model(config: ModelConfig, checkpoint_file: Optional[str] = None) -> torch.nn.Module:
        if config.model_name.startswith('efficientnet'):
            model_builder = modelzoo.efficientnet
        else:
            model_builder = getattr(modelzoo, config.model_name, None)

        if model_builder is None:
            raise ValueError(f'Unexpected model name: {config.model_name}')

        if checkpoint_file is not None:
            config.pretrained = False

        model = model_builder(config)

        if checkpoint_file is not None:
            ckpt = torch.load(checkpoint_file, map_location='cpu')
            state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
            model.load_state_dict(state_dict, strict=True)

        return model
