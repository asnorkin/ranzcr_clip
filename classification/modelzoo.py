import yaml
from torch import nn
from torchvision import models

from efficientnet_pytorch import EfficientNet


class ModelConfig(object):
    def __init__(self, config_file):
        self.params = yaml.safe_load(open(config_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def resnext50_32x4d(num_classes, pretrained=True, **kwargs):
    kwargs['pretrained'] = pretrained

    # Load original model
    model = models.resnext50_32x4d(**kwargs)

    # Change head
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def resnet50(num_classes, pretrained=True, **kwargs):
    kwargs['pretrained'] = pretrained

    # Load original model
    model = models.resnet50(**kwargs)

    # Change head
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def efficientnet_b0(num_classes, pretrained=True):
    params = {
        'model_name': 'efficientnet-b0',
        'num_classes': num_classes,
        'dropout_rate': 0.5,
        'drop_connect_rate': 0.5,
    }

    if pretrained:
        model = EfficientNet.from_pretrained(**params)
    else:
        model = EfficientNet.from_name(**params)

    return model
