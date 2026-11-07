"""Tests for the model wrappers and the datasets that run them."""

from collections.abc import Generator

import pytest
import torch

from src.config import Experiment, get_experiment_config
from src.data import DataSplit
from src.data.derived_datasets import EvaluatedDataset
from src.data.structures import Inputs, Targets
from src.module.classifiers import get_classifier
from src.lib.models import ClassifierModel, VAEEMAModel, VAEModel
from src.module.autoencoders import VAEX

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


@pytest.mark.parametrize('average', [True, False])
def test_vae_reconstructs_with_the_weights_it_evaluates(experiment: Experiment, mocker, average: bool) -> None:
    """The model with an average reconstructs and generates with it; the plain model with the trained weights."""
    model: VAEModel = VAEEMAModel(VAEX(), decay=0.5, name='vaex', device=CPU)
    if not average:
        model.unregister()
        model = VAEModel(VAEX(), name='vaex', device=CPU)

    expected = model.averaged_module if isinstance(model, VAEEMAModel) else model.module
    reconstruct_spy = mocker.spy(expected, 'reconstruct')
    generate_spy = mocker.spy(expected, 'generate')
    inputs = Inputs(image=torch.rand(2, 3, 129, 129), log_probs=torch.full((2, 2), 0.5).log())

    model.reconstruct(inputs, relaxation=0.0)
    model.generate(model.module.get_target_condition(0, 2))

    model.unregister()
    assert reconstruct_spy.call_args.kwargs['relaxation'] == 0.0
    assert generate_spy.call_count == 1


def test_average_follows_the_trained_weights(experiment: Experiment) -> None:
    """After the first update copies the weights, each batch moves the average by the decay."""
    model = VAEEMAModel(VAEX(), decay=0.5, name='vaex', device=CPU)
    model.post_batch_update()
    trained = next(model.module.parameters())
    before = next(model.averaged_module.parameters()).clone()
    with torch.no_grad():
        trained.add_(1.0)

    model.post_batch_update()

    averaged = next(model.averaged_module.parameters())
    model.unregister()
    torch.testing.assert_close(averaged, 0.5 * before + 0.5 * trained)


def _get_running_means(model: VAEModel) -> list[torch.Tensor]:
    return [layer.running_mean.clone() for layer in model.module.modules() if isinstance(layer, torch.nn.BatchNorm2d)]


@torch.inference_mode()
def test_calibration_re_estimates_and_keeps_the_statistics_of_each_path(experiment: Experiment) -> None:
    """Each path gets statistics of its own, and taking a path again restores them."""
    model = VAEModel(VAEX(), name='vaex', device=CPU)
    inputs = Inputs(image=torch.rand(4, 3, 129, 129), log_probs=torch.full((4, 2), 0.5).log())
    model.set_calibration(inputs)
    trained = _get_running_means(model)

    model.reconstruct(inputs)
    reconstruction = _get_running_means(model)
    model.generate(model.module.get_target_condition(0, 2))
    generation = _get_running_means(model)
    model.reconstruct(inputs)
    restored = _get_running_means(model)

    model.unregister()
    assert not torch.equal(reconstruction[0], trained[0])
    assert not torch.equal(generation[-1], reconstruction[-1])
    assert all(torch.equal(a, b) for a, b in zip(restored, reconstruction, strict=True))


@torch.inference_mode()
def test_no_calibration_keeps_the_statistics_of_training(experiment: Experiment) -> None:
    """Without calibration inputs, evaluation uses the statistics that training collected."""
    model = VAEModel(VAEX(), name='vaex', device=CPU)
    inputs = Inputs(image=torch.rand(2, 3, 129, 129), log_probs=torch.full((2, 2), 0.5).log())
    trained = _get_running_means(model)

    model.reconstruct(inputs)

    model.unregister()
    assert all(torch.equal(a, b) for a, b in zip(_get_running_means(model), trained, strict=True))
