import os
import os.path as osp

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.backends import cudnn

from classification.loss import rank_average
from classification.modelzoo import ModelConfig
from classification.module import XRayClassificationModule


class Predictor(object):
    def predict(self, sample):
        return self.predict_batch([sample])[0]

    def predict_batch(self, batch):
        raise NotImplementedError


class TorchModelMixin(object):
    def __init__(self, model, config):
        super().__init__()

        # Save config and model
        self.config = config
        self.model = model

        # Inference mode
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Float
        use_float16 = config.use_float16
        if use_float16 is None:
            use_float16 = torch.cuda.is_available()

        self.float = torch.float16 if use_float16 else torch.float32
        if use_float16:
            self.model = self.model.half()

        # Device
        device = config.device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # Multi GPU
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Cudnn
        if self.device.type == 'cuda':
            cudnn.benchmark = False
            cudnn.deterministic = True
            cudnn.fastest = True

        # Seed
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)


class TorchModelPredictor(TorchModelMixin, Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transform = A.Compose([
            A.Resize(height=self.config.input_height, width=self.config.input_width),
            A.Normalize(max_pixel_value=1.0),
            ToTensorV2(),
        ])

    def predict_batch(self, batch, preprocess=False, output='probabilities', tta=True):
        assert output in {'rank', 'probabilities', 'binary'}

        # Preprocess
        if preprocess:
            batch['image'] = torch.stack([
                self.transform(image=batch['image'][i])['image']
                for i in range(len(batch['image']))
            ])

        # Infer
        batch['image'] = batch['image'].to(self.device).to(self.float)
        predictions = self.model.forward(batch['image'])

        # TTA
        if tta:
            predictions_hflip = self.model.forward(torch.flip(batch['image'], dims=(-1,)))
            if output == 'rank':
                predictions = rank_average(predictions, predictions_hflip)
            else:
                predictions = (predictions + predictions_hflip) / 2

        # Postprocess
        if output != 'rank':
            torch.sigmoid_(predictions)

        if output == 'binary':
            predictions[predictions < self.config.confidence_threshold] = 0
            predictions[predictions >= self.config.confidence_threshold] = 1

        return predictions

    @classmethod
    def create_from_checkpoints(cls, checkpoints_dir):
        config = ModelConfig(osp.join(checkpoints_dir, 'config.yml'))
        model_files = [osp.join(checkpoints_dir, fname) for fname in os.listdir(checkpoints_dir)
                       if fname.startswith('single')]
        if len(model_files) != 1:
            raise RuntimeError(f'Found not unique single model checkpoint: {model_files}')

        model = XRayClassificationModule.build_model(config, model_files[0])
        return cls(model, config)
