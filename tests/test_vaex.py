"""Tests for the published VAEX model and its loss."""

import math
from collections.abc import Generator

import pytest
import torch

from src.config import Experiment, get_experiment_config
from src.data.structures import Inputs, Outputs, Targets
from src.module.autoencoders import VAEX
from src.lib.objectives import GaussianNLL, get_kld_loss


@pytest.fixture(scope='module')
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Generator[Experiment]:
    """Provide an active experiment with the default configuration."""
    exp = Experiment(get_experiment_config(), par_dir=tmp_path_factory.mktemp('vaex'))
    with exp.create_run(record=False):
        yield exp

    return


@pytest.fixture(scope='module')
def module(experiment: Experiment) -> VAEX:
    """Provide the VAE in evaluation mode."""
    return VAEX().eval()


def _center(probability: float) -> float:
    return (math.sqrt(probability) - math.sqrt(1 - probability) + 1) / 2


def _get_inputs(probability_first: float, batch_size: int = 2) -> Inputs:
    probabilities = torch.tensor([[probability_first, 1 - probability_first]]).repeat(batch_size, 1)
    return Inputs(image=torch.rand(batch_size, 3, 129, 129), log_probs=probabilities.log())


class TestVAEX:
    """Test the reconstructions, the counterfactuals and the generation."""

    def test_condition_is_the_centered_first_probability(self, module: VAEX) -> None:
        """The probability of the first class is centered twice."""
        condition = module.condition_from(_get_inputs(0.1))

        torch.testing.assert_close(condition, torch.full((2, 1), _center(_center(0.1))))

    @torch.inference_mode()
    def test_reconstruction_and_posteriors(self, module: VAEX) -> None:
        """The reconstruction has the shape of the image, with a posterior for each latent."""
        outputs = module(_get_inputs(0.3))

        assert outputs.recon.shape == (2, 3, 129, 129)
        assert len(outputs.mu) == len(outputs.log_var) == len(module.z_dims)
        assert len(outputs.prior_log_var) == len(module.z_dims) - 1

    @torch.inference_mode()
    def test_prior_mean_shifts_the_first_channel(self, module: VAEX) -> None:
        """The first channel of the top latent has the scaled condition as prior mean."""
        inputs = _get_inputs(0.3)
        condition = module.condition_from(inputs)

        prior_mu = module(inputs).prior_mu

        first_channel = prior_mu[:, : module.top_channel_dim]
        torch.testing.assert_close(first_channel, (module.prior_scale * condition).expand_as(first_channel))
        assert prior_mu[:, module.top_channel_dim :].abs().max() == 0

    @torch.inference_mode()
    def test_counterfactuals_and_generation(self, module: VAEX) -> None:
        """Counterfactuals and generated samples are images."""
        image = torch.rand(2, 3, 129, 129)

        counterfactual = module.generate_counterfactual(image, target=0, relaxation=0.4).recon
        generated = module.generate(module.get_target_condition(1, 2))

        assert counterfactual.shape == generated.shape == (2, 3, 129, 129)
        assert counterfactual.min() >= 0 and counterfactual.max() <= 1

    def test_no_relaxation_samples_the_lower_latents_from_their_priors(self, module: VAEX) -> None:
        """Without relaxation, the features of the encoder change neither the mean nor the variance."""
        condition = module.condition_from(_get_inputs(0.8))
        encoding = module.encode(torch.rand(2, 3, 129, 129), condition)
        other_hidden = module.encode(torch.rand(2, 3, 129, 129), condition).hidden
        torch.manual_seed(0)
        recon = module.decode(encoding.z, condition, encoding.hidden, relaxation=0.0).recon
        torch.manual_seed(0)

        other_recon = module.decode(encoding.z, condition, other_hidden, relaxation=0.0).recon

        torch.testing.assert_close(recon, other_recon)

    def test_generation_samples_the_top_latent_with_the_others(self, module: VAEX, mocker) -> None:
        """The top latent is sampled like the lower ones, from a standard prior."""
        spy = mocker.spy(module, 'sample')

        module.generate(module.get_target_condition(0, 2))

        top_mean, top_log_var = spy.call_args_list[0].args
        assert torch.equal(top_mean, torch.zeros_like(top_mean))
        assert torch.equal(top_log_var, torch.zeros_like(top_log_var))

    def test_random_condition_is_centered_like_the_classifier(self, module: VAEX) -> None:
        """Random probabilities are turned into a condition as the classifier's are."""
        concentration = torch.ones(module.n_classes)
        torch.manual_seed(0)
        probabilities = torch.distributions.Dirichlet(concentration).sample((64,))
        expected = module.condition_from(Inputs(image=torch.empty(0), log_probs=probabilities.log()))
        torch.manual_seed(0)

        condition = module.get_random_condition(64)

        torch.testing.assert_close(condition, expected)
        assert bool(((condition > 0) & (condition < 1)).any())


class TestGaussianNLL:
    """Test the negative log-likelihood with a learned per-pixel variance."""

    @staticmethod
    def _get_batch(error: float) -> tuple[Outputs, Targets]:
        image = torch.zeros(4, 3, 2, 2)
        outputs = Outputs(recon=image + error, prior_mu=torch.empty(0))
        return outputs, Targets(label=torch.zeros(4), ref=image)

    def test_variance_follows_training_batches_only(self) -> None:
        """The variance moves towards the error of training batches, not of evaluation ones."""
        nll = GaussianNLL(momentum=0.9)
        nll(*self._get_batch(1.0))

        with torch.inference_mode():
            evaluation = nll(*self._get_batch(2.0))['NLL']
        training = nll(*self._get_batch(2.0))['NLL']
        after = nll(*self._get_batch(2.0))['NLL']

        assert evaluation == training
        assert after < training


def test_kld_of_posteriors_equal_to_the_priors(experiment: Experiment) -> None:
    """The divergence is zero and its smooth version is the free bits of every coordinate."""
    zeros = torch.zeros(3, 4)
    outputs = Outputs(
        recon=torch.empty(0),
        prior_mu=zeros,
        mu=[zeros, zeros],
        log_var=[zeros, zeros],
        prior_log_var=[zeros],
    )

    values = get_kld_loss().calculate(outputs, Targets(label=torch.zeros(3)))

    freebits = Experiment.get_config().autoencoder.objective.freebits
    expected_smooth = 0.5 * 8 * (math.log1p(math.exp(-2 * freebits)) + 2 * freebits)
    torch.testing.assert_close(values['KLD'], torch.tensor(0.0))
    torch.testing.assert_close(values['KLD_FB'], torch.tensor(expected_smooth))
