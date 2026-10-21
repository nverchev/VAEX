"""Matplotlib tracker drawing the image grids the events carry."""

import functools
import pathlib

from typing import ClassVar, override

import numpy as np
import torch

from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from drytorch.core import log_events
from drytorch.trackers.base_classes import Dumper

from src.lib.events import ConfidenceEvent, ImageEvent, ImageGridEvent

__all__ = ['ImageTracker']

CELL_SIZE = 3.0
"""Width and height of one image of the grid, in inches."""


class ImageTracker(Dumper):
    """Save every image or image grid an event carries to a file in the run's folder.

    Attributes:
        show: Whether a window also displays the images.
    """

    folder_name: ClassVar[str] = 'images'

    show: bool

    def __init__(self, *, par_dir: pathlib.Path | None = None, show: bool = False) -> None:
        """Initialize.

        Args:
            par_dir: directory the images are filed under.
            show: whether a window also displays the images.
        """
        super().__init__(par_dir=par_dir)
        self.show = show
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: ImageEvent) -> None:
        if not (self.show or self.record):
            return super().notify(event)

        figure = self._new_figure(CELL_SIZE, CELL_SIZE)
        axis = figure.subplots()
        self._draw_image(axis, event.image, event.caption)
        self._finish_figure(figure, event.title, event.folder)
        return super().notify(event)

    @notify.register
    def _(self, event: ImageGridEvent) -> None:
        if not (self.show or self.record):
            return super().notify(event)

        n_rows = len(event.images)
        n_cols = max((len(row) for row in event.images), default=0)
        if not n_cols:
            return super().notify(event)

        figure = self._new_figure(CELL_SIZE * n_cols, CELL_SIZE * n_rows)
        axes = figure.subplots(n_rows, n_cols, squeeze=False)
        for row_axes, row_images, row_captions in zip(axes, event.images, event.captions, strict=True):
            n_images = len(row_images)
            for axis, image, caption in zip(row_axes[:n_images], row_images, row_captions, strict=True):
                self._draw_image(axis, image, caption)

            for axis in row_axes[n_images:]:
                axis.axis('off')

        self._finish_figure(figure, event.title, event.folder)
        return super().notify(event)

    @notify.register
    def _(self, event: ConfidenceEvent) -> None:
        if not (self.show or self.record):
            return super().notify(event)

        probabilities = event.probabilities.detach().cpu().numpy()
        correct = event.correct.detach().cpu().numpy()
        figure = self._new_figure(2 * CELL_SIZE, CELL_SIZE)
        axis = figure.subplots()
        bins = np.linspace(0, 1, 41).tolist()
        axis.hist(
            [probabilities[correct], probabilities[~correct]], bins=bins, stacked=True, label=['correct', 'wrong']
        )
        axis.set_yscale('log')
        axis.set_xlabel('Probability of the predicted class')
        axis.set_ylabel('Images')
        axis.set_title(f'{len(probabilities)} images, {(~correct).sum()} wrong')
        axis.legend()
        self._finish_figure(figure, event.title, event.folder)
        return super().notify(event)

    def _new_figure(self, width: float, height: float) -> Figure:
        return pyplot.figure(figsize=(width, height)) if self.show else Figure(figsize=(width, height))

    def _finish_figure(self, figure: Figure, title: str, folder: str = '') -> None:
        figure.tight_layout()
        if self.record:
            save_dir = self._get_run_dir() / folder
            save_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(save_dir / f'{title}.png', bbox_inches='tight')

        if self.show:
            pyplot.show()
            pyplot.close(figure)

        return

    @staticmethod
    def _draw_image(axis: Axes, image: torch.Tensor, caption: str = '') -> None:
        axis.axis('off')
        pixels = image.detach().cpu().clamp(0, 1)
        if pixels.shape[0] == 1:
            axis.imshow(pixels[0].numpy(), cmap='gray', vmin=0, vmax=1)
        else:
            axis.imshow(pixels.permute(1, 2, 0).numpy())

        if caption:
            axis.set_title(caption)

        return
