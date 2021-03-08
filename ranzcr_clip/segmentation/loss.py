import torch


def dice(inputs, targets, eps=1e-3):
    inputs = torch.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.flatten()
    targets = targets.flatten()

    intersection = (inputs * targets).sum()
    dice = (2.0 * intersection + eps) / (inputs.sum() + targets.sum() + eps)

    return dice


class SegmentationLoss(torch.nn.Module):
    def __init__(self, n_classes, dice_weight=1.0, dice_eps=1e-3, class_weights=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.dice_eps = dice_eps
        self.n_classes = n_classes
        self.class_weights = class_weights

    def bce(self, inputs, targets):
        bce = torch.nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        if self.class_weights is not None:
            bce = bce * self.class_weights

        return bce.mean()

    def forward(self, inputs, targets):
        losses = dict()
        losses['ce'] = self.bce(inputs, targets)
        losses['dice'] = dice(inputs, targets, eps=self.dice_eps)
        losses['total'] = losses['ce'] - self.dice_weight * losses['dice']
        return losses
