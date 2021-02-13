import torch


def dice(inputs, targets, eps=1e-3):
    inputs = torch.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2.0 * intersection + eps) / (inputs.sum() + targets.sum() + eps)

    return dice


class SegmentationLoss(torch.nn.Module):
    def __init__(self, n_classes, dice_weight=1.0, dice_eps=1e-3):
        super().__init__()
        self.dice_weight = dice_weight
        self.dice_eps = dice_eps
        self.n_classes = n_classes

        self.ce = torch.nn.BCEWithLogitsLoss() if n_classes == 1 else torch.nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        losses = dict()
        losses['ce'] = self.ce(inputs, targets)
        losses['dice'] = dice(inputs, targets, eps=self.dice_eps)
        losses['total'] = losses['ce'] - self.dice_weight * losses['dice']
        return losses