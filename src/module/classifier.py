"""Classifier architectures for CelebA facial images."""

import abc
from typing import Final, override

import torch
import torch.nn as nn

from src.config.experiment import Experiment
from src.data.structures import Inputs

__all__ = [
    'BaseClassifier',
    'CelebaClassifier',
    'get_classifier',
]


class BaseClassifier(nn.Module, abc.ABC):
    """Abstract base for all classification model architectures.

    Attributes:
        x_dim: Resolution of input images.
        in_channels: Number of input channels.
        n_classes: Number of target classes to predict.
    """

    x_dim: Final[int]
    in_channels: Final[int]
    n_classes: Final[int]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config()
        self.x_dim = cfg.data.x_dim
        self.in_channels = cfg.data.n_channels
        self.n_classes = len(cfg.data.dataset.classes)
        return

    @abc.abstractmethod
    def forward(self, inputs: Inputs, /) -> torch.Tensor:
        """Forward pass for the classifier."""


class CelebaClassifier(BaseClassifier):
    """5-stage convolutional classifier for CelebA facial images.

    Attributes:
        net: Convolutional layers followed by linear projection to class logits.
    """

    net: Final[nn.Sequential]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        modules = []
        in_channels = self.in_channels
        in_dims = self.x_dim
        for _ in range(5):
            in_dims //= 2
            out_channels = 2 * in_channels
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 1, bias=False),
                    nn.AdaptiveMaxPool2d((in_dims, in_dims)),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.1),
                )
            )
            in_channels = out_channels

        self.net = nn.Sequential(
            *modules,
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_channels * (in_dims**2), self.n_classes),
        )
        return

    @override
    def forward(self, inputs: Inputs, /) -> torch.Tensor:
        return self.net(inputs.image)


def get_classifier() -> BaseClassifier:
    """Instantiate standard CelebA classifier module."""
    return CelebaClassifier()
