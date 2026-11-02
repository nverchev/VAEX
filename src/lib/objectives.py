"""Evaluation metrics and loss specifications."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from drytorch.lib.objectives import DictLoss, Loss, LossWithMetrics, Metric, join

from src.config.experiment import Experiment
from src.data.structures import Outputs, Targets

__all__ = [
    'FreeBitsKLD',
    'GaussianNLL',
    'get_accuracy',
    'get_classifier_loss',
    'get_cross_entropy',
    'get_kld_loss',
    'get_nll_loss',
    'get_vae_loss',
]


class GaussianNLL:
    """Gaussian negative log-likelihood with a fixed standard deviation.

    Attributes:
        sigma: standard deviation of each pixel.
    """

    sigma: float

    def __init__(self, sigma: float) -> None:
        """Initialize.

        Args:
            sigma: standard deviation of each pixel.
        """
        self.sigma = sigma
        return

    def __call__(self, outputs: Outputs, targets: Targets) -> dict[str, torch.Tensor]:
        """Get the negative log-likelihood of the batch and the mean squared error."""
        # mixed precision returns half-precision outputs, too coarse for the loss
        squared_error = F.mse_loss(outputs.recon.float(), targets.ref, reduction='none')
        pixel_nll = 0.5 * squared_error.mean(0) / self.sigma**2 + math.log(self.sigma) + 0.5 * math.log(2 * math.pi)
        return {'NLL': pixel_nll.sum(), 'MSE': squared_error.mean()}


class FreeBitsKLD:
    """KL divergence summed over the latents, smoothed by free bits.

    Each coordinate costs ``softplus(kld - 2 * freebits) + 2 * freebits`` in the loss, which
    stops pushing the divergence below the free bits; ``KLD`` reports the divergence itself.

    Attributes:
        freebits: minimum cost per coordinate below which divergence is not pushed.
    """

    freebits: float

    def __init__(self, freebits: float) -> None:
        """Initialize.

        Args:
            freebits: minimum cost per coordinate below which divergence is not pushed.
        """
        self.freebits = freebits
        return

    def __call__(self, outputs: Outputs, _: Targets) -> dict[str, torch.Tensor]:
        """Calculate the smoothed and raw KL divergence over all latents."""
        divergences = self._calculate_divergences(outputs)
        kld = sum((self._reduce(d) for d in divergences), torch.zeros(()))
        kld_fb = sum((self._reduce(self._smooth(d)) for d in divergences), torch.zeros(()))
        return {'KLD_FB': kld_fb, 'KLD': kld}

    def _calculate_divergences(self, outputs: Outputs) -> list[torch.Tensor]:
        top_divergence = self._top_divergence(
            outputs.mu[0].float(), outputs.log_var[0].float(), outputs.prior_mu.float()
        )
        divergences = [top_divergence]
        stages = zip(outputs.mu[1:], outputs.log_var[1:], outputs.prior_log_var, strict=True)
        for d_mu, d_log_var, prior_log_var in stages:
            divergences.append(self._stage_divergence(d_mu.float(), d_log_var.float(), prior_log_var.float()))

        return divergences

    def _smooth(self, divergence: torch.Tensor) -> torch.Tensor:
        # divergences are 2 * KL; _reduce halves them
        return F.softplus(divergence - 2 * self.freebits) + 2 * self.freebits

    @staticmethod
    def _reduce(divergence: torch.Tensor) -> torch.Tensor:
        return 0.5 * divergence.mean(dim=0).sum()

    @staticmethod
    def _top_divergence(mu: torch.Tensor, log_var: torch.Tensor, prior_mu: torch.Tensor) -> torch.Tensor:
        return -1 - log_var + log_var.exp() + (mu - prior_mu) ** 2

    @staticmethod
    def _stage_divergence(d_mu: torch.Tensor, d_log_var: torch.Tensor, prior_log_var: torch.Tensor) -> torch.Tensor:
        return -1 - d_log_var + d_log_var.exp() + d_mu**2 / prior_log_var.exp()


def get_accuracy() -> Metric[torch.Tensor, Targets]:
    """Return top-1 per-sample accuracy metric."""
    return Metric[torch.Tensor, Targets](_predicts_label, name='Accuracy')


def get_counterfactual_success() -> Metric[torch.Tensor, Targets]:
    """Get the per-sample success of the counterfactuals: the classifier predicts their target class."""
    return Metric[torch.Tensor, Targets](_predicts_label, name='Success')


def get_cross_entropy() -> Loss[torch.Tensor, Targets]:
    """Return cross-entropy loss objective."""
    ce = nn.CrossEntropyLoss(reduction='none')

    def _cross_entropy(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
        return ce(outputs, targets.label)

    return Loss[torch.Tensor, Targets](_cross_entropy, name='CrossEntropy')


def get_classifier_loss() -> LossWithMetrics[torch.Tensor, Targets]:
    """Get the cross entropy, reporting the accuracy as well."""
    return join(get_cross_entropy(), get_accuracy())


def get_nll_loss() -> DictLoss[Outputs, Targets]:
    """Get the negative log-likelihood of the reconstruction, reporting the mean squared error."""
    sigma = Experiment.get_config().autoencoder.objective.sigma
    return DictLoss[Outputs, Targets](GaussianNLL(sigma), name='NLL')


def get_kld_loss() -> DictLoss[Outputs, Targets]:
    """Get the KL divergence summed over the latents, smoothed by free bits.

    Each coordinate costs ``softplus(kld - 2 * freebits) + 2 * freebits`` in the loss, which
    stops pushing the divergence below the free bits; ``KLD`` reports the divergence itself.
    """
    freebits = Experiment.get_config().autoencoder.objective.freebits
    return DictLoss[Outputs, Targets](FreeBitsKLD(freebits), name='KLD_FB')


def get_vae_loss() -> LossWithMetrics[Outputs, Targets]:
    """Get the loss of the VAE: negative log-likelihood and weighted, smoothed KL divergence."""
    c_kld = Experiment.get_config().autoencoder.objective.c_kld
    return get_nll_loss() + c_kld * get_kld_loss()


def _predicts_label(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
    return (outputs.argmax(dim=-1) == targets.label).float()
