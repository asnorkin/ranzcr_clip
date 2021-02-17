from efficientnet_pytorch import EfficientNet

from timm import models as timm_models
from torch import nn
from torchvision import models as tv_models

from common.model_utils import ModelConfig


def resnext50_32x4d(config: ModelConfig, in_channels: int = 1) -> nn.Module:
    params = {
        'pretrained': config.pretrained,
        'num_classes': config.num_classes,
    }

    # Load original model
    model = tv_models.resnext50_32x4d(**params)

    # Change input
    model.conv1.in_channels = in_channels
    model.conv1.weight = nn.Parameter(model.conv1.weight.mean(dim=1, keepdim=True))

    return model


def resnet50(config: ModelConfig, in_channels: int = 1) -> nn.Module:
    params = {
        'pretrained': config.pretrained,
        'num_classes': config.num_classes,
    }

    # Load original model
    model = tv_models.resnet50(**params)

    # Change input
    model.conv1.in_channels = in_channels
    model.conv1.weight = nn.Parameter(model.conv1.weight.mean(dim=1, keepdim=True))

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
    freeze_blocks = config.freeze_blocks or 0
    for i in range(freeze_blocks):
        for p in model._blocks[i].parameters():
            p.requires_grad = False

    return model


def densenet(config: ModelConfig, in_channels: int = 1) -> nn.Module:
    params = {'pretrained': config.pretrained}

    # Load original model
    model = getattr(tv_models, config.model_name)(**params)

    # Change input
    model.features.conv0.in_channels = in_channels
    model.features.conv0.weight = nn.Parameter(model.features.conv0.weight.mean(dim=1, keepdim=True))

    # Change head
    model.classifier = nn.Linear(model.classifier.in_features, config.num_classes)

    return model


def nfnet(config: ModelConfig, in_channels: int = 1) -> nn.Module:
    params = {
        'pretrained': config.pretrained,
        'num_classes': config.num_classes,
        'in_chans': in_channels,
    }

    model = getattr(timm_models.nfnet, config.model_name)(**params)

    return model
