import os
import os.path as osp

import torch

from classification.loss import rank_average
from classification.modelzoo import ModelConfig
from classification.module import XRayClassificationModule
from predictors.predictor import Predictor, TorchModelPredictor


class EnsemblePredictor(Predictor):
    def __init__(self, config, predictors):
        self.config = config
        self.predictors = predictors

    def predict_batch(self, batch, **predict_kwargs):
        batch_predictions = [predictor.predict_batch(batch, **predict_kwargs) for predictor in self.predictors]
        batch_predictions = self.merge(batch_predictions)
        return batch_predictions

    def merge(self, batch_predictions):
        raise NotImplementedError


class FoldPredictor(EnsemblePredictor):
    def merge(self, batch_predictions, output='mean'):
        if output == 'rank':
            return rank_average(batch_predictions)
        elif output == 'mean':
            return torch.stack(batch_predictions).mean(dim=0)
        else:
            raise ValueError(f'Unexpected merge output type: {output}')

    @classmethod
    def create_from_checkpoints(cls, checkpoints_dir):
        config = ModelConfig(osp.join(checkpoints_dir, 'config.yml'))
        predictors = []
        for fname in os.listdir(checkpoints_dir):
            if fname.startswith('fold'):
                model = XRayClassificationModule.build_model(config, osp.join(checkpoints_dir, fname))
                predictors.append(TorchModelPredictor(config=config, model=model))

        return cls(config, predictors)
