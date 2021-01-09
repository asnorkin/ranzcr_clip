import yaml
from torch import nn
from torchvision import models

from efficientnet_pytorch import EfficientNet


class ModelConfig:
    def __init__(self, config_file: str):
        self.params = yaml.safe_load(open(config_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


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


def efficientnet(config: ModelConfig) -> nn.Module:
    params = {
        'model_name': config.model_name.replace('_', '-'),
        'in_channels': 1,
        'num_classes': config.num_classes,
        'dropout_rate': config.dropout_rate,
        'drop_connect_rate': config.drop_connect_rate,
        'width_coefficient': config.width_coefficient,
    }

    if config.pretrained:
        model = EfficientNet.from_pretrained(**params)
    else:
        model = EfficientNet.from_name(**params)

    return model
