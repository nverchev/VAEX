"""Weights & Biases tracker logging the images the events carry."""

import functools

from typing import override

import wandb

from drytorch.core import log_events
from drytorch.trackers.wandb import Wandb

from src.train.events import ConfidenceEvent, ImageEvent, ImageGridEvent

__all__ = ['ImageWandb']


class ImageWandb(Wandb):
    """Weights & Biases tracker that also logs the images and the image grids."""

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: ImageEvent) -> None:
        if self.record:
            image = wandb.Image(event.image.detach().cpu().clamp(0, 1), caption=event.caption or None)
            self.run.log({self._get_key(event.title, event.folder): image})

        return super().notify(event)

    @notify.register
    def _(self, event: ImageGridEvent) -> None:
        if self.record:
            # each row of the grid is a gallery, so the cells of the rows line up by position
            key = self._get_key(event.title, event.folder)
            galleries = {
                f'{key}/row_{i}': [
                    wandb.Image(image.detach().cpu().clamp(0, 1), caption=caption)
                    for image, caption in zip(row_images, row_captions, strict=True)
                ]
                for i, (row_images, row_captions) in enumerate(zip(event.images, event.captions, strict=True))
                if row_images
            }
            self.run.log(galleries)

        return super().notify(event)

    @notify.register
    def _(self, event: ConfidenceEvent) -> None:
        if self.record:
            key = self._get_key(event.title, event.folder)
            probabilities = event.probabilities.detach().cpu().numpy()
            correct = event.correct.detach().cpu().numpy()
            histograms = {
                f'{key}/correct': wandb.Histogram(probabilities[correct].tolist()),
                f'{key}/wrong': wandb.Histogram(probabilities[~correct].tolist()),
            }
            self.run.log(histograms)

        return super().notify(event)

    @staticmethod
    def _get_key(title: str, folder: str) -> str:
        return f'{folder}/{title}' if folder else title
