"""Evaluation metrics and loss specifications."""

from __future__ import annotations

import math

from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from drytorch.lib.objectives import DictLoss, Loss, LossWithMetrics, Metric, join

from src.config.experiment import Experiment
from src.data.structures import Outputs, Targets

__all__ = [
    'AdversarialKLD',
    'FreeBitsKLD',
    'GaussianNLL',
    'KLDAnnealing',
    'get_accuracy',
    'get_adversarial_loss',
    'get_annealing',
    'get_classifier_loss',
    'get_cross_entropy',
    'get_kld_loss',
    'get_nll_loss',
    'get_vae_loss',
]


class GaussianNLL:
    """Gaussian negative log-likelihood with a fixed variance.

    Attributes:
        sigma_2: variance of each pixel.
    """

    sigma_2: float

    def __init__(self, sigma_2: float) -> None:
        """Initialize.

        Args:
            sigma_2: variance of each pixel.
        """
        self.sigma_2 = sigma_2
        return

    def __call__(self, outputs: Outputs, targets: Targets) -> dict[str, torch.Tensor]:
        """Get the negative log-likelihood of the batch and the mean squared error."""
        # mixed precision returns half-precision outputs, too coarse for the loss
        squared_error = F.mse_loss(outputs.recon.float(), targets.ref, reduction='none')
        normalization = 0.5 * math.log(self.sigma_2) + 0.5 * math.log(2 * math.pi)
        pixel_nll = 0.5 * squared_error.mean(0) / self.sigma_2 + normalization
        return {'NLL': pixel_nll.sum(), 'MSE': squared_error.mean()}


class FreeBitsKLD:
    """KL divergence summed over the latents, smoothed by the free bits of each stage.

    A coordinate of a stage with free bits ``fb`` costs ``softplus(kld - 2 * fb) + offset`` in
    the loss, where the offset makes a coordinate with no divergence cost exactly ``fb``. The
    loss stops pushing the divergence below the free bits. ``KLD`` reports the divergence
    itself, and ``KLD_stage_<i>`` the divergence of each stage, from the top latent down.
    ``Active_stage_<i>`` reports the fraction of the coordinates of each stage whose divergence,
    averaged over the batch, exceeds ``ACTIVE_THRESHOLD``: it falls to zero when a stage collapses.

    Attributes:
        freebits: free bits of each coordinate of each stage, from the top latent down.
    """

    ACTIVE_THRESHOLD: Final = 0.01

    freebits: tuple[float, ...]

    def __init__(self, freebits: tuple[float, ...]) -> None:
        """Initialize.

        Args:
            freebits: free bits of each coordinate of each stage, from the top latent down.
        """
        self.freebits = freebits
        return

    def __call__(self, outputs: Outputs, _: Targets) -> dict[str, torch.Tensor]:
        """Calculate the smoothed and raw KL divergence over all latents, and of each stage."""
        divergences = self._calculate_divergences(outputs)
        kld = torch.zeros(())
        kld_fb = torch.zeros(())
        stage_metrics: dict[str, torch.Tensor] = {}
        for index, (divergence, freebits) in enumerate(zip(divergences, self.freebits, strict=True)):
            stage_kld = self._reduce(divergence)
            kld = kld + stage_kld
            kld_fb = kld_fb + self._reduce(self._smooth(divergence, freebits))
            stage_metrics[f'KLD_stage_{index}'] = stage_kld
            coordinate_kld = 0.5 * divergence.mean(dim=0)
            stage_metrics[f'Active_stage_{index}'] = (coordinate_kld > self.ACTIVE_THRESHOLD).float().mean()

        return {'KLD_FB': kld_fb, 'KLD': kld, **stage_metrics}

    def _calculate_divergences(self, outputs: Outputs) -> list[torch.Tensor]:
        stages = zip(outputs.mu, outputs.log_var, outputs.prior_log_var, strict=True)
        return [
            self._divergence(d_mu.float(), d_log_var.float(), p_log_var.float())
            for d_mu, d_log_var, p_log_var in stages
        ]

    @staticmethod
    def _smooth(divergence: torch.Tensor, freebits: float) -> torch.Tensor:
        # divergences are 2 * KL; _reduce halves them
        offset = 2 * freebits - math.log1p(math.exp(-2 * freebits))
        return F.softplus(divergence - 2 * freebits) + offset

    @staticmethod
    def _reduce(divergence: torch.Tensor) -> torch.Tensor:
        return 0.5 * divergence.mean(dim=0).sum()

    @staticmethod
    def _divergence(d_mu: torch.Tensor, d_log_var: torch.Tensor, prior_log_var: torch.Tensor) -> torch.Tensor:
        # twice the KL divergence of a posterior shifted by (d_mu, d_log_var) from its prior
        return -1 - d_log_var + d_log_var.exp() + d_mu**2 / prior_log_var.exp()


class KLDAnnealing:
    """Cyclical weight of the KL divergence.

    In each cycle of the annealing period, the weight rises along a cosine from ``MIN_WEIGHT`` to
    one, then restarts, so that the reconstruction can bring back latents the KL divergence
    began to close. After the period, the weight is one.

    Attributes:
        period: epochs of the annealing; ``0`` keeps the weight at one.
        cycles: cycles in the period.
    """

    MIN_WEIGHT: Final = 0.01

    period: int
    cycles: int

    def __init__(self, period: int, cycles: int) -> None:
        """Initialize.

        Args:
            period: epochs of the annealing; ``0`` keeps the weight at one.
            cycles: cycles in the period.
        """
        self.period = period
        self.cycles = cycles
        return

    def __call__(self, outputs: Outputs, _: Targets) -> torch.Tensor:
        """Get the weight at the epoch of the outputs, for each sample."""
        progress = outputs.epoch / self.period if self.period else 1.0
        weight = 1.0
        if progress < 1:
            phase = (progress * self.cycles) % 1
            weight = max(0.5 * (1 - math.cos(math.pi * phase)), self.MIN_WEIGHT)

        return torch.full((len(outputs.recon),), weight, device=outputs.recon.device)


class AdversarialKLD:
    """KL divergence of the predictions of the adversary from the probabilities of the classifier.

    The adversary minimizes it and, through the reversed gradient, the free channels of the top
    latent maximize it. Its weight rises from zero to one over the warm-up epochs.

    Attributes:
        warmup_epochs: epochs of the warm-up; ``0`` starts at full weight.
    """

    warmup_epochs: int

    def __init__(self, warmup_epochs: int) -> None:
        """Initialize.

        Args:
            warmup_epochs: epochs of the warm-up; ``0`` starts at full weight.
        """
        self.warmup_epochs = warmup_epochs
        return

    def __call__(self, outputs: Outputs, _: Targets) -> torch.Tensor:
        """Get the weighted divergence of each sample."""
        adv_log_probs = F.log_softmax(outputs.adv_logits.float(), dim=1)
        divergence = F.kl_div(adv_log_probs, outputs.log_probs.float(), reduction='none', log_target=True).sum(1)
        scale = min(1.0, outputs.epoch / self.warmup_epochs) if self.warmup_epochs else 1.0
        return scale * divergence


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
    sigma_2 = Experiment.get_config().autoencoder.objective.sigma_2
    return DictLoss[Outputs, Targets](GaussianNLL(sigma_2), name='NLL')


def get_kld_loss() -> DictLoss[Outputs, Targets]:
    """Get the KL divergence summed over the latents, smoothed by the free bits of each stage.

    See :class:`FreeBitsKLD` for the cost of each coordinate and the reported metrics.
    """
    freebits = Experiment.get_config().autoencoder.objective.freebits
    return DictLoss[Outputs, Targets](FreeBitsKLD(freebits), name='KLD_FB')


def get_annealing() -> Loss[Outputs, Targets]:
    """Get the cyclical weight of the KL divergence; see :class:`KLDAnnealing`."""
    train_cfg = Experiment.get_config().autoencoder.train
    return Loss[Outputs, Targets](
        KLDAnnealing(train_cfg.annealing_period, train_cfg.annealing_cycles), name='Annealing'
    )


def get_adversarial_loss() -> Loss[Outputs, Targets]:
    """Get the loss of the adversary on the free channels of the top latent; see :class:`AdversarialKLD`."""
    warmup_epochs = Experiment.get_config().autoencoder.train.adv_warmup_epochs
    return Loss[Outputs, Targets](AdversarialKLD(warmup_epochs), name='Adv')


def get_vae_loss() -> LossWithMetrics[Outputs, Targets]:
    """Get the loss of the VAE: negative log-likelihood, annealed KL divergence and adversarial loss."""
    return get_nll_loss() + get_kld_loss() * get_annealing() + get_adversarial_loss()


def _predicts_label(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
    return (outputs.argmax(dim=-1) == targets.label).float()
