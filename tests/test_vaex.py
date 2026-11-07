"""Tests for the published VAEX model and its loss."""

import math
from collections.abc import Generator

import pytest
import torch

from src.config import Experiment, get_experiment_config
from src.data.structures import Inputs, Outputs, Targets
from src.module.autoencoders import VAEX
from src.lib.objectives import FreeBitsKLD, GaussianNLL


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


def _get_inputs(probability_first: float, batch_size: int = 2) -> Inputs:
    probabilities = torch.tensor([[probability_first, 1 - probability_first]]).repeat(batch_size, 1)
    return Inputs(image=torch.rand(batch_size, 3, 129, 129), log_probs=probabilities.log())


class TestVAEX:
    """Test the reconstructions, the counterfactuals and the generation."""

    def test_condition_is_the_tempered_softmax(self, module: VAEX) -> None:
        """The log-probabilities of the classifier are divided by the temperature before the softmax."""
        inputs = _get_inputs(0.1)
        expected = torch.softmax(inputs.log_probs / module.temperature, dim=1)

        condition = module.condition_from(inputs)

        torch.testing.assert_close(condition, expected)

    @torch.inference_mode()
    def test_reconstruction_and_posteriors(self, module: VAEX) -> None:
        """The reconstruction has the shape of the image, with a posterior for each latent."""
        outputs = module(_get_inputs(0.3))

        assert outputs.recon.shape == (2, 3, 129, 129)
        assert len(outputs.mu) == len(outputs.log_var) == len(module.z_dims)
        assert len(outputs.prior_log_var) == len(module.z_dims) - 1

    @torch.inference_mode()
    def test_prior_mean_shifts_the_conditioned_channels(self, module: VAEX) -> None:
        """The conditioned channels of the top latent have the learned prior mean, the others zero."""
        inputs = _get_inputs(0.3)
        condition = module.condition_from(inputs)

        prior_mu = module(inputs).prior_mu

        torch.testing.assert_close(prior_mu[:, : module.conditioned_dim], module.condition_to_prior(condition))
        assert prior_mu[:, module.conditioned_dim :].abs().max() == 0

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

    def test_random_condition_is_a_probability_vector(self, module: VAEX) -> None:
        """Random conditions are probabilities over all the classes."""
        condition = module.get_random_condition(64)

        assert condition.shape == (64, module.n_classes)
        torch.testing.assert_close(condition.sum(dim=1), torch.ones(64))


class TestGaussianNLL:
    """Test the negative log-likelihood with a fixed standard deviation."""

    @staticmethod
    def _get_batch(error: float) -> tuple[Outputs, Targets]:
        image = torch.zeros(4, 3, 2, 2)
        outputs = Outputs(recon=image + error, prior_mu=torch.empty(0))
        return outputs, Targets(label=torch.zeros(4), ref=image)

    def test_perfect_reconstruction_costs_the_normalization_only(self) -> None:
        """Without error, the value is the Gaussian normalization of each pixel."""
        nll = GaussianNLL(sigma=0.5)

        value = nll(*self._get_batch(0.0))['NLL']

        assert value.item() == pytest.approx(12 * (math.log(0.5) + 0.5 * math.log(2 * math.pi)))

    def test_error_is_weighted_by_the_inverse_variance(self) -> None:
        """Halving sigma multiplies the cost of the squared error by four."""
        wide, narrow = GaussianNLL(sigma=1.0), GaussianNLL(sigma=0.5)

        wide_cost = wide(*self._get_batch(1.0))['NLL'] - wide(*self._get_batch(0.0))['NLL']
        narrow_cost = narrow(*self._get_batch(1.0))['NLL'] - narrow(*self._get_batch(0.0))['NLL']

        assert narrow_cost.item() == pytest.approx(4 * wide_cost.item())

    def test_training_does_not_change_the_loss(self) -> None:
        """The value is the same before and after a training batch."""
        nll = GaussianNLL(sigma=0.5)

        before = nll(*self._get_batch(1.0))['NLL']
        nll(*self._get_batch(2.0))
        after = nll(*self._get_batch(1.0))['NLL']

        assert before == after


def test_kld_of_posteriors_equal_to_the_priors() -> None:
    """The divergence is zero, and its smooth version is the free bits of every coordinate of each stage."""
    zeros = torch.zeros(3, 4)
    outputs = Outputs(
        recon=torch.empty(0),
        prior_mu=zeros,
        mu=[zeros, zeros],
        log_var=[zeros, zeros],
        prior_log_var=[zeros],
    )

    values = FreeBitsKLD((0.5, 0.25))(outputs, Targets(label=torch.zeros(3)))

    torch.testing.assert_close(values['KLD'], torch.tensor(0.0))
    torch.testing.assert_close(values['KLD_stage_0'], torch.tensor(0.0))
    torch.testing.assert_close(values['KLD_stage_1'], torch.tensor(0.0))
    torch.testing.assert_close(values['KLD_FB'], torch.tensor(4 * 0.5 + 4 * 0.25))


def test_freebits_must_match_the_stages() -> None:
    """A configuration with free bits for fewer stages than the latents raises ValueError."""
    with pytest.raises(ValueError, match='freebits'):
        get_experiment_config(overrides=['autoencoder.objective.freebits=[1.0,0.5]'])


def test_adam_with_weight_decay_is_rejected() -> None:
    """Adam with weight decay raises ValueError: the weight decay needs AdamW."""
    overrides = ['autoencoder.train.learn.optimizer=Adam', 'autoencoder.train.learn.opt_settings.weight_decay=0.01']

    with pytest.raises(ValueError, match='AdamW'):
        get_experiment_config(overrides=overrides)
