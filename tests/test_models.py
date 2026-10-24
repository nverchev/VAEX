"""Tests for the model wrappers and the datasets that run them."""

from collections.abc import Generator

import pytest
import torch

from src.config import Experiment, get_experiment_config
from src.data import DataSplit
from src.data.derived_datasets import EvaluatedDataset
from src.data.structures import Inputs, Targets
from src.module.classifiers import get_classifier
from src.lib.models import ClassifierModel

CPU = torch.device('cpu')


class _Images(DataSplit):
    def __init__(self, n_images: int) -> None:
        self.n_images = n_images
        return

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        return Inputs(image=torch.rand(3, 129, 129)), Targets(label=torch.tensor(index))

    def __len__(self) -> int:
        return self.n_images


@pytest.fixture(scope='module')
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Generator[Experiment]:
    """Provide an active experiment with the default configuration."""
    exp = Experiment(get_experiment_config(), par_dir=tmp_path_factory.mktemp('models'))
    with exp.create_run(record=False):
        yield exp

    return


@pytest.fixture(scope='module')
def classifier(experiment: Experiment) -> ClassifierModel:
    """Provide a classifier model."""
    return ClassifierModel(get_classifier(), name='test_classifier', device=CPU)


def test_log_probabilities(classifier: ClassifierModel) -> None:
    """The log-probabilities of each image sum to one in probability."""
    log_probs = classifier.log_probabilities(Inputs(image=torch.rand(2, 3, 129, 129)))

    torch.testing.assert_close(log_probs.exp().sum(dim=-1), torch.ones(2))


def test_evaluated_dataset_runs_the_model(classifier: ClassifierModel) -> None:
    """The dataset attaches the log-probabilities of the model to each sample."""
    dataset = EvaluatedDataset(_Images(2), classifier)

    items = dataset.__getitems__([0, 1])

    assert len(items) == 2
    assert all(inputs.log_probs.dtype == torch.float32 for inputs, _ in items)
