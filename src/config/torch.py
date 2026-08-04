"""Torch config utilities."""

import math
import random

import numpy as np
import torch
import torch.nn as nn

type ActClass = type[nn.Module]
type NormClass = type[nn.Module]


class GroupedNorm(nn.GroupNorm):
    """GroupNorm subclass that automatically computes the optimal number of groups for num_channels."""

    def __init__(self, num_channels: int) -> None:
        n_groups = int(math.sqrt(num_channels))
        while num_channels % n_groups != 0:
            n_groups -= 1

        super().__init__(num_groups=n_groups, num_channels=num_channels)
        return


DEFAULT_ACT: ActClass = nn.ELU
DEFAULT_NORM: NormClass = nn.BatchNorm2d


def get_activation_cls(act_name: str) -> ActClass:
    """Get activation class from name."""
    try:
        return getattr(nn.modules.activation, act_name)
    except AttributeError as ae:
        raise ValueError(f'Input act_name "{act_name}" is not the name of a pytorch activation.') from ae


def get_norm_cls(norm_name: str) -> NormClass:
    """Get normalization class from name."""
    if norm_name == 'GroupedNorm':
        return GroupedNorm

    try:
        return getattr(nn, norm_name)
    except AttributeError as ae:
        raise ValueError(f'Input norm_name "{norm_name}" is not the name of a pytorch nn.Module.') from ae


def get_optim_cls(optimizer_name: str) -> type[torch.optim.Optimizer]:
    """Get optimizer class from name."""
    try:
        return getattr(torch.optim, optimizer_name)
    except AttributeError as ae:
        raise ValueError(f'Input opt_name "{optimizer_name}" is not the name of a pytorch optimizer.') from ae


def set_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return None
