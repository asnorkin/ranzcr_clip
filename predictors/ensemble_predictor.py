import os
import os.path as osp

import torch

from classification.model import ModelConfig
from classification.module import XRayClassificationModule
from predictors.predictor import Predictor, TorchModelPredictor


class EnsemblePredictor(Predictor):
    def __init__(self, predictors):
        self.predictors = predictors

    def predict_batch(self, batch):
        batch_predictions = [predictor.predict_batch(batch) for predictor in self.predictors]
        batch_predictions = self.merge(batch_predictions)
        return batch_predictions

    def merge(self, batch_predictions):
        raise NotImplementedError


class FoldPredictor(EnsemblePredictor):
    def merge(self, batch_predictions):
        return torch.cat(batch_predictions).mean(dim=0)

    @classmethod
    def create_from_checkpoints(cls, checkpoints_dir):
        config = ModelConfig(osp.join(checkpoints_dir, 'config.yml'))
        predictors = []
        for fname in os.listdir(checkpoints_dir):
            if fname.startswith('fold'):
                predictors.append(
                    TorchModelPredictor(
                        config=config,
                        model=XRayClassificationModule(config)))

        return cls(predictors)
