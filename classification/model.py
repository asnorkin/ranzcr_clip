import yaml
from torch import nn
from torchvision import models


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
