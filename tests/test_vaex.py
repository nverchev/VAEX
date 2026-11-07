"""Tests for the published VAEX model and its loss."""

import math
from collections.abc import Generator

import pytest
import torch

from src.config import Experiment, get_experiment_config
from src.data.structures import Encoding, Inputs, Outputs, Targets
from src.module.autoencoders import VAEX
from src.lib.objectives import AdversarialKLD, FreeBitsKLD, GaussianNLL, KLDAnnealing
from src.module.layers import GradientReversal


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
        """The reconstruction has the shape of the image, with a posterior and a prior for each latent."""
        inputs = _get_inputs(0.3)

        outputs = module(inputs)

        assert outputs.recon.shape == (2, 3, 129, 129)
        assert len(outputs.mu) == len(outputs.log_var) == len(outputs.prior_log_var) == len(module.z_dims)
        assert outputs.adv_logits.shape == (2, module.n_classes)
        assert torch.equal(outputs.log_probs, inputs.log_probs)

    @torch.inference_mode()
    def test_only_the_conditioned_channels_have_a_prior_from_the_condition(self, module: VAEX) -> None:
        """The prior of the conditioned channels changes with the condition; the one of the free channels does not."""
        first = module.condition_from(_get_inputs(0.1))
        second = module.condition_from(_get_inputs(0.9))

        first_mu, first_log_var = module.get_top_prior(first)
        second_mu, second_log_var = module.get_top_prior(second)

        conditioned = slice(None, module.conditioned_dim)
        free = slice(module.conditioned_dim, None)
        assert not torch.equal(first_mu[:, conditioned], second_mu[:, conditioned])
        assert torch.equal(first_mu[:, free], second_mu[:, free])
        assert torch.equal(first_log_var[:, free], second_log_var[:, free])

    @torch.inference_mode()
    def test_counterfactuals_and_generation(self, module: VAEX) -> None:
        """Counterfactuals and generated samples are images."""
        inputs = _get_inputs(0.3)

        counterfactual = module.generate_counterfactual(inputs, target=0, relaxation=0.4).recon
        generated = module.generate(module.get_target_condition(1, 2))

        assert counterfactual.shape == generated.shape == (2, 3, 129, 129)
        assert counterfactual.min() >= 0 and counterfactual.max() <= 1

    @torch.inference_mode()
    def test_counterfactuals_sample_the_conditioned_channels_from_the_prior_of_the_target(
        self, module: VAEX, mocker
    ) -> None:
        """The conditioned channels of the top latent ignore their posterior; the free ones keep it."""
        inputs = _get_inputs(0.3)
        target_condition = module.get_target_condition(0, 2)
        prior_mu, _ = module.get_top_prior(target_condition)
        encoding = module.encode(inputs.image, target_condition)
        spy = mocker.spy(module, 'sample')

        module.generate_counterfactual(inputs, target=0, relaxation=1.0)

        top_mean = spy.call_args_list[0].args[0]
        conditioned = slice(None, module.conditioned_dim)
        free = slice(module.conditioned_dim, None)
        torch.testing.assert_close(top_mean[:, conditioned], prior_mu[:, conditioned])
        torch.testing.assert_close(top_mean[:, free], prior_mu[:, free] + encoding.mu[:, free])

    @torch.inference_mode()
    def test_strength_mixes_the_condition_with_the_target(self, module: VAEX, mocker, monkeypatch) -> None:
        """The condition of the counterfactuals weighs the target class by the strength."""
        inputs = _get_inputs(0.3)
        monkeypatch.setattr(module, 'strength', 0.25)
        expected = 0.75 * module.condition_from(inputs) + 0.25 * module.get_target_condition(1, 2)
        spy = mocker.spy(module, 'encode')

        module.generate_counterfactual(inputs, target=1, relaxation=1.0)

        torch.testing.assert_close(spy.call_args.args[1], expected)

    @torch.inference_mode()
    def test_no_relaxation_samples_the_lower_latents_from_their_priors(self, module: VAEX) -> None:
        """Without relaxation, the features of the encoder change neither the mean nor the variance."""
        condition = module.condition_from(_get_inputs(0.8))
        encoding = module.encode(torch.rand(2, 3, 129, 129), condition)
        other_hidden = module.encode(torch.rand(2, 3, 129, 129), condition).hidden
        other_encoding = Encoding(mu=encoding.mu, log_var=encoding.log_var, hidden=other_hidden)
        torch.manual_seed(0)
        recon = module.decode(condition, encoding, relaxation=0.0).recon
        torch.manual_seed(0)

        other_recon = module.decode(condition, other_encoding, relaxation=0.0).recon

        torch.testing.assert_close(recon, other_recon)

    @torch.inference_mode()
    def test_generation_samples_the_top_latent_from_its_prior(self, module: VAEX, mocker) -> None:
        """Without an encoding, the top latent is sampled from the prior of the condition."""
        condition = module.get_target_condition(0, 2)
        prior_mu, prior_log_var = module.get_top_prior(condition)
        spy = mocker.spy(module, 'sample')

        module.generate(condition)

        top_mean, top_log_var = spy.call_args_list[0].args
        torch.testing.assert_close(top_mean, prior_mu)
        torch.testing.assert_close(top_log_var, prior_log_var)

    def test_evaluation_scales_the_standard_deviation_of_each_latent(self, module: VAEX) -> None:
        """In evaluation mode, the noise of a latent is scaled by its factor in sampling_std."""
        zeros = torch.zeros(1000)
        torch.manual_seed(0)
        noise = torch.randn(1000)
        torch.manual_seed(0)

        sample = module.sample(zeros, zeros, level=2)

        torch.testing.assert_close(sample, module.sampling_std[2] * noise)

    def test_posterior_weights_grow_towards_the_pixels(self, module: VAEX) -> None:
        """The finest latent has the relaxation as its weight, each coarser one a further factor of it."""
        weights = module.get_posterior_weights(0.5)

        assert weights == (0.5**5, 0.5**4, 0.5**3, 0.5**2, 0.5)

    def test_random_condition_is_a_probability_vector(self, module: VAEX) -> None:
        """Random conditions are probabilities over all the classes."""
        condition = module.get_random_condition(64)

        assert condition.shape == (64, module.n_classes)
        torch.testing.assert_close(condition.sum(dim=1), torch.ones(64))


class TestGaussianNLL:
    """Test the negative log-likelihood with a fixed variance."""

    @staticmethod
    def _get_batch(error: float) -> tuple[Outputs, Targets]:
        image = torch.zeros(4, 3, 2, 2)
        return Outputs(recon=image + error), Targets(label=torch.zeros(4), ref=image)

    def test_perfect_reconstruction_costs_the_normalization_only(self) -> None:
        """Without error, the value is the Gaussian normalization of each pixel."""
        nll = GaussianNLL(sigma_2=0.25)

        value = nll(*self._get_batch(0.0))['NLL']

        assert value.item() == pytest.approx(12 * (0.5 * math.log(0.25) + 0.5 * math.log(2 * math.pi)))

    def test_error_is_weighted_by_the_inverse_variance(self) -> None:
        """Dividing the variance by four multiplies the cost of the squared error by four."""
        wide, narrow = GaussianNLL(sigma_2=1.0), GaussianNLL(sigma_2=0.25)

        wide_cost = wide(*self._get_batch(1.0))['NLL'] - wide(*self._get_batch(0.0))['NLL']
        narrow_cost = narrow(*self._get_batch(1.0))['NLL'] - narrow(*self._get_batch(0.0))['NLL']

        assert narrow_cost.item() == pytest.approx(4 * wide_cost.item())


def test_kld_of_posteriors_equal_to_the_priors() -> None:
    """The divergence is zero, and its smooth version is the free bits of every coordinate of each stage."""
    zeros = torch.zeros(3, 4)
    outputs = Outputs(recon=torch.empty(0), mu=[zeros, zeros], log_var=[zeros, zeros], prior_log_var=[zeros, zeros])

    values = FreeBitsKLD((0.5, 0.25))(outputs, Targets(label=torch.zeros(3)))

    torch.testing.assert_close(values['KLD'], torch.tensor(0.0))
    torch.testing.assert_close(values['KLD_stage_0'], torch.tensor(0.0))
    torch.testing.assert_close(values['KLD_stage_1'], torch.tensor(0.0))
    torch.testing.assert_close(values['KLD_FB'], torch.tensor(4 * 0.5 + 4 * 0.25))


def test_active_coordinates_are_those_above_the_threshold() -> None:
    """A stage reports the fraction of its coordinates whose divergence exceeds the threshold."""
    shift = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    zeros = torch.zeros(1, 4)
    outputs = Outputs(recon=torch.empty(0), mu=[shift], log_var=[zeros], prior_log_var=[zeros])

    values = FreeBitsKLD((0.0,))(outputs, Targets(label=torch.zeros(1)))

    torch.testing.assert_close(values['Active_stage_0'], torch.tensor(0.25))


def test_kld_of_a_shifted_mean_is_scaled_by_the_prior_variance() -> None:
    """A shift of the mean costs half its square divided by the prior variance."""
    shift = torch.full((1, 1), 2.0)
    zeros = torch.zeros(1, 1)
    outputs = Outputs(
        recon=torch.empty(0), mu=[shift], log_var=[zeros], prior_log_var=[torch.full((1, 1), math.log(4))]
    )

    values = FreeBitsKLD((0.0,))(outputs, Targets(label=torch.zeros(1)))

    torch.testing.assert_close(values['KLD'], torch.tensor(0.5))


def test_freebits_must_match_the_stages() -> None:
    """A configuration with free bits for fewer stages than the latents raises ValueError."""
    with pytest.raises(ValueError, match='freebits'):
        get_experiment_config(overrides=['autoencoder.objective.freebits=[1.0,0.5]'])


def test_adam_with_weight_decay_is_rejected() -> None:
    """Adam with weight decay raises ValueError: the weight decay needs AdamW."""
    overrides = ['autoencoder.train.learn.optimizer=Adam', 'autoencoder.train.learn.opt_settings.weight_decay=0.01']

    with pytest.raises(ValueError, match='AdamW'):
        get_experiment_config(overrides=overrides)


class TestKLDAnnealing:
    """Test the cyclical weight of the KL divergence."""

    @staticmethod
    def _get_weight(annealing: KLDAnnealing, epoch: int) -> float:
        outputs = Outputs(recon=torch.zeros(2, 1), epoch=epoch)
        return annealing(outputs, Targets(label=torch.zeros(2)))[0].item()

    def test_each_cycle_starts_at_the_minimum_and_rises_along_a_cosine(self) -> None:
        """The weight is the minimum at the start of a cycle and one half in its middle."""
        annealing = KLDAnnealing(period=90, cycles=2)

        start = self._get_weight(annealing, 0)
        middle = self._get_weight(annealing, 22)
        restart = self._get_weight(annealing, 45)

        assert start == restart == pytest.approx(KLDAnnealing.MIN_WEIGHT)
        assert middle == pytest.approx(0.5 * (1 - math.cos(math.pi * 44 / 90)))

    def test_weight_is_one_after_the_period_or_without_one(self) -> None:
        """After the period, and when the period is zero, the weight is one."""
        after = self._get_weight(KLDAnnealing(period=90, cycles=2), 90)
        disabled = self._get_weight(KLDAnnealing(period=0, cycles=2), 0)

        assert after == disabled == 1.0


class TestAdversarialKLD:
    """Test the loss of the adversary."""

    @staticmethod
    def _get_outputs(adv_probability: float, epoch: int) -> Outputs:
        adv_logits = torch.tensor([[adv_probability, 1 - adv_probability]]).log()
        return Outputs(
            recon=torch.zeros(1), adv_logits=adv_logits, log_probs=torch.tensor([[0.5, 0.5]]).log(), epoch=epoch
        )

    def test_matching_predictions_cost_nothing(self) -> None:
        """When the adversary predicts the probabilities of the classifier, the divergence is zero."""
        loss = AdversarialKLD(warmup_epochs=0)

        value = loss(self._get_outputs(0.5, epoch=0), Targets(label=torch.zeros(1)))

        torch.testing.assert_close(value, torch.zeros(1))

    def test_weight_rises_over_the_warmup(self) -> None:
        """Halfway through the warm-up, the divergence counts half."""
        targets = Targets(label=torch.zeros(1))

        full = AdversarialKLD(warmup_epochs=0)(self._get_outputs(0.9, epoch=1), targets)
        half = AdversarialKLD(warmup_epochs=2)(self._get_outputs(0.9, epoch=1), targets)

        torch.testing.assert_close(half, 0.5 * full)


def test_gradient_reversal_flips_and_scales_the_gradient() -> None:
    """The forward pass is the identity; the gradient comes back multiplied by minus the strength."""
    x = torch.ones(3, requires_grad=True)

    y = GradientReversal.apply(x, 0.5)
    y.sum().backward()

    assert torch.equal(y.detach(), torch.ones(3))
    assert torch.equal(x.grad, torch.full((3,), -0.5))
