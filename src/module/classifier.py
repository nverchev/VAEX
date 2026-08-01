"""Classifier architectures for CelebA facial images."""

import abc
import torch
import torch.nn as nn

from src.config.experiment import Experiment
from src.data.structures import Inputs


class BaseClassifier(nn.Module, abc.ABC):
    """Base classifier class."""

    num_classes: int

    @abc.abstractmethod
    def forward(self, inputs: Inputs) -> torch.Tensor:
        """Forward pass for the classifier."""


class CelebaClassifier(BaseClassifier):
    """5-stage convolutional classifier for CelebA facial images."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        x_dim = cfg.data.x_dim
        n_classes = len(cfg.data.dataset.classes)
        modules = []
        in_channels = cfg.data.n_channels
        in_dims = x_dim
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
            nn.Linear(in_channels * (in_dims**2), n_classes),
        )
        return

    def forward(self, inputs: Inputs) -> torch.Tensor:
        return self.net(inputs.image )

def get_classifier() -> BaseClassifier:
    """Instantiate standard CelebA classifier module."""
    return CelebaClassifier()
