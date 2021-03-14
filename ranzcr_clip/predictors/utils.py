from typing import List, Tuple, Union

import torch


def logits(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    min_ = min(tensor.min().item(), eps)
    max_ = max(tensor.max().item(), 1.0 - eps)
    tensor = (tensor - (tensor.min() - min_)) / (max_ - min_)
    return torch.log(tensor / (1.0 - tensor))


def reduce_mean(tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], power: float = 1.0) -> torch.Tensor:
    assert len(tensors) > 0

    # No reduction
    if len(tensors) == 1:
        return tensors[0]

    result = torch.stack(tensors)

    # Simple mean
    if power == 1.0:
        return result.mean(dim=0)

    # Rescale negative values to [0, 1] because non integer power will return nan on negative values
    need_rescale = result.min() < 0
    if need_rescale:
        result.sigmoid_()

    # Power mean
    result = torch.pow(result, power).mean(dim=0)
    result = torch.pow(result, 1.0 / power)

    # Rescale values back
    if need_rescale:
        result = logits(result)

    return result


def rank_average(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.argsort(dim=1).argsort(dim=1).float().mean(dim=0) / tensor.shape[1]
