import numpy as np
from pytorch_lightning.metrics.functional.classification import auroc

import torch
from torch import nn
from torch.nn import functional as F


def linear_combination(x, y, alpha):
    return alpha * x + (1 - alpha) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()

    return arr


def reduce_auc_roc(auc_roc_values, reduction='mean'):
    if reduction is None:
        return auc_roc_values

    result = auc_roc_values[auc_roc_values != -1]
    if len(result) == 0:
        print(f'No targets has valid ROC AUC, result is zero.')
        return torch.tensor(0.).to(auc_roc_values)

    return reduce_loss(result, reduction=reduction)


def batch_auc_roc(targets, probabilities, reduction='mean'):
    N_TARGETS = 11

    result = torch.ones(N_TARGETS).to(targets) * -1
    for i in range(11):
        targets_i, probabilities_i = targets[:, i], probabilities[:, i]
        if torch.unique(targets_i).numel() == 1:
            print(f'Target {i} has only one class. Skip it in ROC AUC.')
        else:
            result[i] = auroc(probabilities_i, targets_i)

    return reduce_auc_roc(result, reduction=reduction)


def rank_average(*tensors):
    assert len(tensors) > 1

    norm = len(tensors) * len(tensors[0])

    result = torch.zeros_like(tensors[0])
    for tensor in tensors:
        result += torch.argsort(tensor, dim=0) / norm

    return result


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits, targets):
        n = logits.size()[-1]
        log_preds = F.log_softmax(logits, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, targets, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, epsilon=0.0, clip=0.0, weights=None, reduction='mean'):
        super().__init__()

        assert clip >= 0.0
        self.clip = clip
        self._clip_logit_lo = torch.logit(torch.tensor(clip)) if clip != 0.0 else None
        self._clip_logit_hi = torch.logit(1. - torch.tensor(clip)) if clip != 0.0 else None

        self.epsilon = epsilon
        self.weights = weights
        self.reduction = reduction

    def forward(self, logits, targets):
        # Smooth labels
        targets = (1 - 2 * self.epsilon) * targets + self.epsilon

        # Clip logits
        if self.clip != 0.0:
            logits = torch.clamp(logits, self._clip_logit_lo, self._clip_logit_hi)

        # Compute bce
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, weight=self.weights, reduction=self.reduction)

        return bce

    @staticmethod
    def calculate_weights(targets):
        targets = torch.from_numpy(np.stack(targets))
        positive_frac = targets.sum(dim=0) / targets.shape[0]
        negative_frac = 1. - positive_frac
        weights = negative_frac / positive_frac
        weights[torch.isinf(weights) | torch.isnan(weights)] = 1.
        weights = torch.sqrt(weights)
        return weights
