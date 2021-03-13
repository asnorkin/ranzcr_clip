import os
import os.path as osp
from typing import Optional

import torch

from classification.modelzoo import ModelConfig
from classification.module import XRayClassificationModule
from predictors.predictor import Predictor, TorchModelPredictor
from predictors.utils import reduce_mean


class EnsemblePredictor(Predictor):
    def __init__(self, config: ModelConfig, predictors: list):
        self.config = config
        self.predictors = predictors

    def predict_batch(self, batch, reduction: str = 'mean', power: float = 1.0, **predict_kwargs) -> torch.Tensor:
        batch_predictions = [predictor.predict_batch(batch, **predict_kwargs) for predictor in self.predictors]
        batch_predictions = self.merge(batch_predictions, reduction=reduction, power=power)
        return batch_predictions

    def merge(self, batch_predictions: list, reduction: str = 'mean', power: float = 1) -> torch.Tensor:
        raise NotImplementedError


class FoldPredictor(EnsemblePredictor):
    def merge(self, batch_predictions: list, reduction: str = 'mean', power: float = 1.0) -> torch.Tensor:
        if reduction not in {'none', 'mean'}:
            raise ValueError(f'Unexpected merge reduction type: {reduction}')

        if reduction == 'none':
            return torch.stack(batch_predictions)

        return reduce_mean(batch_predictions, power=power)

    @classmethod
    def create_from_checkpoints(cls, checkpoints_dir: str, folds: Optional[str] = None):
        if folds is not None:
            folds = set(map(int, folds))

        config = ModelConfig(osp.join(checkpoints_dir, 'config.yml'))
        predictors = []
        for fname in os.listdir(checkpoints_dir):
            if fname.startswith('fold'):
                fold = int(fname[4:5])
                if folds is not None and fold not in folds:
                    continue

                model = XRayClassificationModule.build_model(config, osp.join(checkpoints_dir, fname))
                predictors.append(TorchModelPredictor(config=config, model=model))

        return cls(config, predictors)
