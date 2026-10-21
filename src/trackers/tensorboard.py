"""TensorBoard tracker writing the images the events carry."""

import functools

from collections.abc import Sequence
from typing import override

import torch

from torchvision.utils import make_grid

from drytorch.core import log_events
from drytorch.trackers.tensorboard import TensorBoard

from src.lib.events import ConfidenceEvent, ImageEvent, ImageGridEvent

__all__ = ['ImageTensorBoard']


class ImageTensorBoard(TensorBoard):
    """TensorBoard tracker that also writes the images and the image grids."""

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: ImageEvent) -> None:
        if not self.record:
            return super().notify(event)

        name = self._get_tag(event.title, event.folder)
        self.writer.add_image(name, event.image.detach().cpu().clamp(0, 1))
        if event.caption:
            self.writer.add_text(name, event.caption)

        return super().notify(event)

    @notify.register
    def _(self, event: ImageGridEvent) -> None:
        n_cols = max((len(row) for row in event.images), default=0)
        if not self.record or not n_cols:
            return super().notify(event)

        name = self._get_tag(event.title, event.folder)
        blank = torch.zeros_like(next(image for row in event.images for image in row))
        cells: list[torch.Tensor] = []
        for row in event.images:
            # short rows are padded, so that each row of the event is a row of the grid
            cells.extend([*row, *[blank] * (n_cols - len(row))])

        grid = make_grid([cell.detach().cpu().clamp(0, 1) for cell in cells], nrow=n_cols, pad_value=1)
        self.writer.add_image(name, grid)
        self.writer.add_text(name, self._markdown(event.captions, n_cols))
        return super().notify(event)

    @notify.register
    def _(self, event: ConfidenceEvent) -> None:
        if self.record:
            name = self._get_tag(event.title, event.folder)
            probabilities = event.probabilities.detach().cpu()
            correct = event.correct.detach().cpu()
            self.writer.add_histogram(f'{name}/correct', probabilities[correct], bins='auto')
            self.writer.add_histogram(f'{name}/wrong', probabilities[~correct], bins='auto')

        return super().notify(event)

    @staticmethod
    def _get_tag(title: str, folder: str) -> str:
        return f'{folder}/{title}' if folder else title

    @staticmethod
    def _markdown(captions: Sequence[Sequence[str]], n_cols: int) -> str:
        # the captions as a table with the layout of the grid
        header = '|' + ' |' * n_cols
        separator = '|---' * n_cols + '|'
        rows = ['| ' + ' | '.join([*row, *[''] * (n_cols - len(row))]) + ' |' for row in captions]
        return '\n'.join([header, separator, *rows])
