"""Tests for the Fréchet Inception Distance."""

from typing import Any, override

import numpy as np
import pytest
import torch
from drytorch.core import protocols as p
from pytest_mock import MockerFixture
from scipy import linalg
from torch import nn

from src.train import fid as fid_module
from src.train.fid import FrechetInceptionDistance

CPU = torch.device('cpu')


@pytest.fixture
def batches() -> list[torch.Tensor]:
    """Provide three batches of small images."""
    generator = torch.Generator().manual_seed(0)
    return [torch.rand(16, 1, 2, 2, generator=generator) for _ in range(3)]


@pytest.fixture
def model(mocker: MockerFixture) -> p.ModelProtocol[Any, Any]:
    """Provide a model named vae at epoch 7."""
    model = mocker.create_autospec(p.ModelProtocol, instance=True)
    model.name = 'vae'
    model.epoch = 7
    return model


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Provide the metric events the distance publishes, recorded instead of dispatched."""
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(fid_module.log_events, 'MetricEvent', lambda **kwargs: events.append(kwargs))
    return events


class _FlatFeatures(nn.Module):
    """Features that are the flattened images, which avoids downloading the network."""

    num_features: int = 4

    @override
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.flatten(1)


def _get_frechet_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    x, y = first.flatten(1).double().numpy(), second.flatten(1).double().numpy()
    sigma_x, sigma_y = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    covmean = np.real(linalg.sqrtm(sigma_x @ sigma_y))
    diff = x.mean(0) - y.mean(0)
    return float(diff @ diff + np.trace(sigma_x + sigma_y - 2 * covmean))


class TestFrechetInceptionDistance:
    """Test the distance between generated and reference images."""

    def test_identical_images_have_no_distance(
        self, batches: list[torch.Tensor], model: p.ModelProtocol[Any, Any], published: list[dict[str, Any]]
    ) -> None:
        """Images compared with themselves are at distance zero."""
        fid = FrechetInceptionDistance(batches, device=CPU, features=_FlatFeatures())

        distance = fid(model, batches)

        assert distance == pytest.approx(0, abs=1e-6)

    def test_distance_is_the_frechet_distance_of_the_features(
        self, batches: list[torch.Tensor], model: p.ModelProtocol[Any, Any], published: list[dict[str, Any]]
    ) -> None:
        """The distance matches the Fréchet distance of the Gaussians fitted to the features."""
        fid = FrechetInceptionDistance(batches[:2], device=CPU, features=_FlatFeatures())
        generated = batches[2] * 0.5

        distance = fid(model, [generated])

        assert distance == pytest.approx(_get_frechet_distance(generated, torch.cat(batches[:2])), rel=1e-5)

    def test_reference_is_kept_between_calls(
        self, batches: list[torch.Tensor], model: p.ModelProtocol[Any, Any], published: list[dict[str, Any]]
    ) -> None:
        """Each call compares only its own images with the reference."""
        fid = FrechetInceptionDistance(batches[:2], device=CPU, features=_FlatFeatures())
        first = fid(model, [batches[2] * 0.5])

        second = fid(model, [batches[2] * 0.5])

        assert second == pytest.approx(first)

    def test_distance_is_published_as_a_metric(
        self, batches: list[torch.Tensor], model: p.ModelProtocol[Any, Any], published: list[dict[str, Any]]
    ) -> None:
        """The distance is published with its name and the name and epoch of the model."""
        fid = FrechetInceptionDistance(batches[:2], device=CPU, features=_FlatFeatures())

        distance = fid(model, [batches[2]], name='FID_test')

        assert published == [{'model_name': 'vae', 'source_name': 'FID_test', 'epoch': 7, 'metrics': {'FID': distance}}]
