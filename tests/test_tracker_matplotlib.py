"""Tests for the tracker saving image grids."""

from collections.abc import Sequence
import pathlib

import pytest
import torch
from matplotlib import pyplot

from src.config import Experiment, get_experiment_config
from src.trackers.matplotlib import ImageTracker
from src.lib.events import ImageEvent, ImageGridEvent


def _publish(
    tmp_path: pathlib.Path,
    images: Sequence[Sequence[torch.Tensor]],
    captions: Sequence[Sequence[str]],
    title: str,
    folder: str = '',
    record: bool = True,
    show: bool = False,
) -> list[pathlib.Path]:
    exp = Experiment(get_experiment_config(), par_dir=tmp_path)
    exp.trackers.subscribe(ImageTracker(show=show))
    with exp.create_run(record=record):
        ImageGridEvent(images=images, captions=captions, title=title, folder=folder)

    return sorted(tmp_path.rglob('*.png'))


def test_a_grid_is_saved_under_its_title_and_folder(tmp_path: pathlib.Path) -> None:
    """The grid becomes one image, named by its title, in the folder it asks for."""
    image = torch.rand(3, 8, 8)

    saved = _publish(
        tmp_path,
        images=[[image, image]],
        captions=[['a', 'b']],
        title='grid',
        folder='sample_0',
    )

    assert [path.name for path in saved] == ['grid.png']
    assert saved[0].parent.name == 'sample_0'
    assert saved[0].parents[1].name != 'sample_0'


def test_a_grey_image_is_drawn(tmp_path: pathlib.Path) -> None:
    """An image with a single channel is drawn in grey."""
    saved = _publish(tmp_path, images=[[torch.rand(1, 8, 8)]], captions=[['digit']], title='grey')

    assert [path.name for path in saved] == ['grey.png']


def test_rows_of_different_lengths_are_saved(tmp_path: pathlib.Path) -> None:
    """A row shorter than the others leaves its last cells blank."""
    image = torch.rand(3, 8, 8)

    saved = _publish(
        tmp_path,
        images=[[image, image, image], [image] * 5],
        captions=[['a', 'b', 'c'], ['r'] * 5],
        title='uneven',
    )

    assert [path.name for path in saved] == ['uneven.png']


def test_a_shown_grid_is_also_saved(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tracker that shows the images still saves them for a recorded run."""
    shown: list[bool] = []
    monkeypatch.setattr(pyplot, 'show', lambda: shown.append(True))
    image = torch.rand(3, 8, 8)

    saved = _publish(tmp_path, images=[[image]], captions=[['a']], title='grid', show=True)

    assert shown == [True]
    assert [path.name for path in saved] == ['grid.png']


def test_an_unrecorded_run_saves_nothing(tmp_path: pathlib.Path) -> None:
    """A grid published in an unrecorded run is not saved."""
    image = torch.rand(3, 8, 8)

    saved = _publish(tmp_path, images=[[image]], captions=[['a']], title='grid', record=False)

    assert saved == []


def test_a_single_image_event_is_saved(tmp_path: pathlib.Path) -> None:
    """A single image event is saved with its caption."""
    exp = Experiment(get_experiment_config(), par_dir=tmp_path)
    exp.trackers.subscribe(ImageTracker())
    image = torch.rand(3, 8, 8)
    with exp.create_run():
        ImageEvent(image=image, title='single', caption='Sample 0', folder='gen')

    saved = sorted(tmp_path.rglob('*.png'))
    assert [path.name for path in saved] == ['single.png']
    assert saved[0].parent.name == 'gen'
