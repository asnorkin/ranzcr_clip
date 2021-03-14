import os
import os.path as osp
from typing import Dict, List, Union

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import LightningModule
from torch.backends import cudnn
from torchvision.transforms.functional import resize

from classification.module import XRayClassificationModule
from common.model_utils import ModelConfig
from predictors.utils import reduce_mean


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
        self,
        batch: Union[Dict, List],
        preprocess: bool = False,
        output: str = 'probabilities',
        tta: bool = True,
        tta_reduction: str = 'mean',
        tta_reduction_power: float = 1.0,
    ) -> torch.Tensor:

        assert tta_reduction in {'mean', 'none'}
        assert output in {'logits', 'probabilities', 'binary'}
        if output != 'logits':
            assert tta_reduction == 'mean'

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
            if output == 'logits' and tta_reduction == 'none':
                logits = torch.stack((logits, logits_hflip))
            else:
                # We need to flip image for segmentation
                if len(logits.shape) > 2:
                    logits_hflip = torch.flip(logits_hflip, dims=(-1,))

                logits = reduce_mean((logits, logits_hflip), power=tta_reduction_power)

        if output == 'logits':
            return logits

        # Postprocess
        predictions = torch.sigmoid(logits)

        if output == 'binary':
            predictions[predictions < self.config.confidence_threshold] = 0
            predictions[predictions >= self.config.confidence_threshold] = 1
            predictions = predictions.long()

        # Segmentation masks postprocess
        if len(predictions.shape) > 2:
            # Prepare masks
            predictions = (predictions * 255).long()  # .transpose(0, 2, 3, 1)
            if predictions.shape[-1] == 1:
                predictions = predictions[..., 0]

            # Resize to the original size
            predictions = torch.stack(
                [resize(prediction, size=batch['ori_shape'][i]) for i, prediction in enumerate(predictions)]
            )

            # Fix resize error
            predictions[predictions < 128] = 0
            predictions[predictions >= 128] = 255

            # Fix dtype and dimensions
            predictions = predictions.type(torch.uint8).permute(0, 2, 3, 1)

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
