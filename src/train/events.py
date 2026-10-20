"""Events published during training and evaluation."""

from __future__ import annotations

import dataclasses

from collections.abc import Sequence

import torch

from drytorch.core import log_events

__all__ = ['ConfidenceEvent', 'ImageEvent', 'ImageGridEvent']


@dataclasses.dataclass(frozen=True)
class ConfidenceEvent(log_events.Event):
    """Probability of the class the classifier predicts for each image, to show as a distribution.

    Attributes:
        probabilities: the probability of the predicted class of each image, of shape ``(N,)``.
        correct: whether each prediction is correct, of shape ``(N,)``.
        title: the name of the distribution, which names its file.
        folder: the subfolder of the run's images, empty for none.
    """

    probabilities: torch.Tensor
    correct: torch.Tensor
    title: str
    folder: str = ''


@dataclasses.dataclass(frozen=True)
class ImageEvent(log_events.Event):
    """Single image to save.

    Attributes:
        image: the image ``(C, H, W)`` with one or three channels and values in ``[0, 1]``.
        title: the name of the image, which names its file.
        caption: optional caption for the image.
        folder: the subfolder of the run's images, empty for none.
    """

    image: torch.Tensor
    title: str
    caption: str = ''
    folder: str = ''


@dataclasses.dataclass(frozen=True)
class ImageGridEvent(log_events.Event):
    """Images to show as a grid, row by row.

    Attributes:
        images: the rows of the grid, each image ``(C, H, W)`` with one or
            three channels and values in ``[0, 1]``.
        captions: one caption per cell, with the same rows as ``images``.
        title: the name of the grid, which names its file.
        folder: the subfolder of the run's images, empty for none.
    """

    images: Sequence[Sequence[torch.Tensor]]
    captions: Sequence[Sequence[str]]
    title: str
    folder: str = ''
