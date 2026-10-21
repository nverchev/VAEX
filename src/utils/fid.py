"""Fréchet Inception Distance between generated images and reference images."""

import logging

from collections.abc import Callable, Iterable
from typing import Any, Final

import torch

from drytorch.core import log_events
from drytorch.core import protocols as p
from torch import Tensor, nn
from torchmetrics.image import fid

__all__ = [
    'FrechetInceptionDistance',
]


class _TorchmetricsFID(fid.FrechetInceptionDistance):
    # torchmetrics replaces update and compute on the instance, which hides their signatures
    compute: Callable[[], Tensor]
    update: Callable[[Tensor, bool], None]


class FrechetInceptionDistance:
    """Fréchet Inception Distance of generated images from reference images.

    The features of the reference images are computed once, when the distance is created.
    """

    _device: Final[torch.device]
    _metric: Final[_TorchmetricsFID]

    def __init__(
        self,
        reference: Iterable[Tensor],
        *,
        device: torch.device,
        features: int | nn.Module = 2048,
    ) -> None:
        """Initialize.

        Args:
            reference: batches of reference images with values in [0, 1].
            device: device of the Inception network.
            features: number of pool features of the Inception network, or a module computing
                other features.
        """
        self._device = device
        self._metric = _TorchmetricsFID(features, reset_real_features=False, normalize=True).to(device)
        for batch in reference:
            self._update(batch, real=True)

        return

    def __call__(self, model: p.ModelProtocol[Any, Any], images: Iterable[Tensor], *, name: str = 'FID') -> float:
        """Get the distance of the images that a model generated, and publish it.

        Args:
            model: the model that generated the images.
            images: batches of generated images with values in [0, 1].
            name: name under which the distance is published.

        Returns:
            The distance.

        Side Effects:
            Publishes the distance as a metric of the model at its current epoch.
        """
        logging.info('Calculating %s for %s.', name, model.name)
        self._metric.reset()
        for batch in images:
            self._update(batch, real=False)

        distance = self._metric.compute().item()
        log_events.MetricEvent(model_name=model.name, source_name=name, epoch=model.epoch, metrics={'FID': distance})
        return distance

    def _update(self, images: Tensor, *, real: bool) -> None:
        # the metric turns values into bytes, where values outside [0, 1] would wrap around
        self._metric.update(images.to(self._device).float().clamp(0, 1), real)
        return
