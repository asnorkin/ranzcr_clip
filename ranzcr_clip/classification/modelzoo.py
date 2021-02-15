from efficientnet_pytorch import EfficientNet

from torch import nn
from torchvision import models

from common.model_utils import ModelConfig


def resnext50_32x4d(config: ModelConfig) -> nn.Module:
    params = {'pretrained': config.pretrained}

    # Load original model
    model = models.resnext50_32x4d(**params)

    # Change head
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.num_classes)

    return model


def resnet50(config: ModelConfig) -> nn.Module:
    params = {'pretrained': config.pretrained}

    # Load original model
    model = models.resnet50(**params)

    # Change head
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.num_classes)

    return model


def efficientnet(config: ModelConfig, in_channels: int = 1) -> nn.Module:
    # Set up params
    params = {
        'model_name': config.model_name.replace('_', '-'),
        'in_channels': in_channels,
        'num_classes': config.num_classes,
        'dropout_rate': config.dropout_rate,
        'drop_connect_rate': config.drop_connect_rate,
        'width_coefficient': config.width_coefficient,
    }

    if config.weights_path is not None:
        params['weights_path'] = config.weights_path

    # Load pretrained weights
    if config.pretrained:
        model = EfficientNet.from_pretrained(**params)
    else:
        model = EfficientNet.from_name(**params)

    # Freeze first blocks
    for i in range(getattr(config, 'freeze_blocks', 0)):
        for p in model._blocks[i].parameters():
            p.requires_grad = False

    return model
