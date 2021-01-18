import torch


def dice(inputs, targets, smooth=1.0):
    inputs = torch.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return dice
