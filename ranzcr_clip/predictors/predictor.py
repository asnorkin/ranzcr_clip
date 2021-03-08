import os
import os.path as osp
from typing import Dict, List, Union

import albumentations as A
import albumentations.augmentations.functional as AF
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import LightningModule
from torch.backends import cudnn

from classification.module import XRayClassificationModule
from common.model_utils import ModelConfig


class Predictor:
    def predict(self, sample: Union[Dict, np.ndarray], *args, **kwargs):
        if isinstance(sample, dict):
            batch = {key: [val] for key, val in sample.items()}
        elif isinstance(sample, np.ndarray):
            batch = {'image': [sample]}
        else:
            raise TypeError(f'Unexpected type of sample: {type(sample)}')

        return self.predict_batch(batch, *args, **kwargs)[0]

    def predict_batch(self, batch: Union[Dict, List], *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class TorchModelMixin:
    def __init__(self, model: torch.nn.Module, config: ModelConfig):
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

        self.transform = A.Compose(
            [
                A.Resize(height=self.config.input_height, width=self.config.input_width, always_apply=True),
                A.CLAHE(always_apply=True),
                A.Normalize(mean=0.482, std=0.220, always_apply=True),  # Ranzcr
                ToTensorV2(always_apply=True),
            ]
        )

    def predict_batch(
        self, batch: Union[Dict, List], preprocess: bool = False, output: str = 'probabilities', tta: bool = True
    ) -> torch.Tensor:
        assert output in {'logits', 'probabilities', 'binary'}

        if isinstance(batch, list):
            batch = {'image': batch}

        # Save original shapes
        batch['ori_shape'] = [image.shape for image in batch['image']]

        # Preprocess
        if preprocess:
            batch['image'] = torch.stack(
                [self.transform(image=batch['image'][i])['image'] for i in range(len(batch['image']))]
            )

        # Infer
        batch['image'] = batch['image'].to(self.device).to(self.float)
        logits = self.model.forward(batch['image'])

        # TTA
        if tta:
            logits_hflip = self.model.forward(torch.flip(batch['image'], dims=(-1,)))
            if output == 'logits':
                logits = logits, logits_hflip
            else:
                logits = (logits + logits_hflip) / 2

        if output == 'logits':
            return logits

        # Postprocess
        predictions = torch.sigmoid(logits).cpu().numpy()

        if output == 'binary':
            predictions[predictions < self.config.confidence_threshold] = 0
            predictions[predictions >= self.config.confidence_threshold] = 1
            predictions = predictions.astype(int)

        # Segmentation masks postprocess
        if len(predictions.shape) > 2:
            # Prepare masks
            predictions = (predictions * 255).astype(np.uint8).transpose(0, 2, 3, 1)
            if predictions.shape[-1] == 1:
                predictions = predictions[..., 0]

            # Resize to the original size
            resized_predictions = []
            for i, prediction in enumerate(predictions):
                ori_height, ori_width = batch['ori_shape'][i]
                resized_predictions.append(AF.resize(prediction, height=ori_height, width=ori_width))
            predictions = np.stack(resized_predictions)

            # Fix resize error
            predictions[predictions < 128] = 0
            predictions[predictions >= 128] = 255

        return predictions

    @classmethod
    def create_from_checkpoints(cls, checkpoints_dir: str, module_class: LightningModule = XRayClassificationModule):
        config = ModelConfig(osp.join(checkpoints_dir, 'config.yml'))
        model_files = [
            osp.join(checkpoints_dir, fname) for fname in os.listdir(checkpoints_dir) if fname.endswith('.ckpt')
        ]

        if len(model_files) == 0:
            raise RuntimeError('Model file not found.')

        if len(model_files) != 1:
            raise RuntimeError(f'Found non unique single model checkpoint: {model_files}')

        model = module_class.build_model(config, model_files[0])
        return cls(model, config)
