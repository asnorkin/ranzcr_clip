from typing import List, Tuple, Union

import torch


def reduce_mean(tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], power: float = 1.0) -> torch.Tensor:
    assert len(tensors) > 0

    # No reduction
    if len(tensors) == 1:
        return tensors[0]

    tensors = torch.stack(tensors)

    # Simple mean
    if power == 1.0:
        return tensors.mean(dim=0)

    # Power mean
    result = torch.pow(tensors, power).mean(dim=0)
    result = torch.pow(result, 1.0 / power)

    return result


def rank_average(tensor: torch.Tensor) -> torch.Tensor:
    return torch.argsort(tensor, dim=1).float().mean(dim=0) / tensor.shape[1]
