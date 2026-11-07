"""Tests of the preprocessing of a single image."""

import pathlib

from collections.abc import Generator

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import Experiment
from src.config.hydra import get_experiment_config
from src.data.preprocess import get_preprocessor


@pytest.fixture(scope='module')
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Generator[Experiment]:
    """Provide an active experiment with the default configuration."""
    exp = Experiment(get_experiment_config(), par_dir=tmp_path_factory.mktemp('preprocess'))
    with exp.create_run(record=False):
        yield exp

    return


def test_load_image_gives_an_8_bit_image_of_the_dataset_size(experiment: Experiment, tmp_path: pathlib.Path) -> None:
    """A loaded image has the size of the dataset images and values on the 8-bit grid."""
    pixels = np.random.default_rng(0).integers(0, 256, (218, 178, 3), dtype=np.uint8)
    path = tmp_path / 'face.png'
    Image.fromarray(pixels).save(path)
    size = experiment.config.data.x_dim

    image = get_preprocessor().load_image(path)

    assert image.shape == (3, size, size)
    assert torch.equal(image.mul(255).round(), image.mul(255))
    assert 0 <= image.min() and image.max() <= 1
