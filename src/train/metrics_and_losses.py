"""Evaluation metrics and loss specifications."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Sequence

from drytorch.lib.objectives import DictLoss, Loss, LossBase, Metric
from src.config.experiment import Experiment
from src.data.structures import Outputs, Targets


def get_accuracy() -> Metric[torch.Tensor | Outputs, Targets]:
    """Return top-1 per-sample accuracy metric."""

    def _accuracy(outputs: torch.Tensor | Outputs, targets: Targets) -> torch.Tensor:
        logits = outputs.logits if isinstance(outputs, Outputs) else outputs

        return (logits.argmax(dim=-1) == targets.label).float()

    return Metric(_accuracy, name='Accuracy')


def get_cross_entropy() -> LossBase[torch.Tensor | Outputs, Targets]:
    """Return cross-entropy loss objective."""
    ce = nn.CrossEntropyLoss(reduction='none')

    def _cross_entropy(outputs: torch.Tensor | Outputs, targets: Targets) -> torch.Tensor:
        logits = outputs.logits if isinstance(outputs, Outputs) else outputs
        return ce(logits, targets.label)

    return Loss(_cross_entropy, name='CrossEntropy')


class NLLLoss:
    """Callable class for Gaussian negative log-likelihood reconstruction loss component."""

    def __init__(self, sigma: float = 0.1) -> None:
        self.adj_nll: float = 0.5 * math.log(2 * math.pi)
        self.var: float = sigma ** 2
        self.logvar: float = math.log(self.var)
        return

    def __call__(self, outputs: Outputs, targets: Targets) -> dict[str, torch.Tensor]:
        ref_image = targets.ref
        SE = F.mse_loss(outputs.recon, ref_image, reduction='none')
        NLL = 0.5 * (SE / self.var + self.logvar).mean(0).sum() + self.adj_nll * SE[0].numel()
        MSE = SE.mean()
        return {'NLL': NLL, 'MSE': MSE}


def get_nll_loss() -> LossBase[Outputs, Targets]:
    """Get Gaussian negative log-likelihood reconstruction loss component with MSE metric."""
    cfg = Experiment.get_config()
    objective_cfg = cfg.autoencoder.objective
    return DictLoss(NLLLoss(sigma=objective_cfg.nll_sigma), name='NLL')


class KLDLoss:
    """Callable class for KL divergence loss, with a free-bits floor per latent stage."""

    def __init__(self, freebits: Sequence[float]) -> None:
        self.freebits = freebits
        return

    @staticmethod
    def _free_bits(matrix: torch.Tensor, freebits: float) -> torch.Tensor:
        offset = 2 * freebits - F.softplus(torch.tensor(-2 * freebits))
        return F.softplus(matrix - 2 * freebits) + offset

    def __call__(self, outputs: Outputs, _: Targets) -> dict[str, torch.Tensor]:
        # p_mu + d_mu = mu, p_logvar + d_logvar = logvar
        kld = torch.zeros((), device=outputs.recon.device)
        kld_real = torch.zeros((), device=outputs.recon.device)
        metrics: dict[str, torch.Tensor] = {}
        zipped = zip(outputs.d_mu, outputs.d_logvar, outputs.prior_log_var, self.freebits, strict=False)
        for i, (d_mu, d_logvar, p_logvar, stage_freebits) in enumerate(zipped):
            stage_matrix = -1 - d_logvar + d_logvar.exp() + (d_mu**2) * torch.exp(-p_logvar)
            stage_free_bits = self._free_bits(stage_matrix, stage_freebits)
            stage_kld_fb = 0.5 * stage_free_bits.mean(0).sum()
            stage_kld = 0.5 * stage_matrix.mean(0).sum()
            kld = kld + stage_kld_fb
            kld_real = kld_real + stage_kld
            metrics[f'KLD_stage_{i}'] = stage_kld

        metrics['KLD_FB'] = kld
        metrics['KLD'] = kld_real
        return metrics


def get_kld_loss() -> LossBase[Outputs, Targets]:
    """Get KL divergence loss component (free bits floor per latent stage)."""
    cfg = Experiment.get_config()
    return DictLoss(KLDLoss(freebits=cfg.autoencoder.model.freebits), name='KLD_FB')


def get_annealing() -> LossBase[Outputs, Targets]:
    """Get (reverse) annealing component matching SurfaceVAE."""
    cfg = Experiment.get_config()
    annealing_period = cfg.autoencoder.train.annealing_period

    min_weight = 0.01

    def _annealing(outputs: Outputs, _: Targets) -> torch.Tensor:
        device = outputs.recon.device
        time_fraction = torch.tensor(outputs.model_epoch / annealing_period, device=device)
        time_fraction = torch.clamp(time_fraction, 0.0, 1.0)
        weight = 0.5 * (1.0 - torch.cos(time_fraction * math.pi))
        return torch.clamp(weight, min=min_weight)

    return Loss(_annealing, name='Annealing')


def get_vae_loss() -> LossBase[Outputs, Targets]:
    """Get VAE loss composed with drytorch natural syntax."""
    cfg = Experiment.get_config()
    nll = get_nll_loss()
    kld = get_kld_loss()
    if cfg.autoencoder.train.annealing_period > 0:
        return nll + kld * get_annealing()

    return nll + kld


def get_classifier_loss() -> LossBase:
    """Get main loss objective based on active experiment configuration."""
    loss = get_cross_entropy()
    loss.watch(get_accuracy())
    return loss
