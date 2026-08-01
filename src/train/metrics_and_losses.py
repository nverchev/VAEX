"""Evaluation metrics and loss specifications."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self) -> None:
        self.adj_nll: float = float(0.5 * np.log(2 * np.pi))
        self.train_var: torch.Tensor | None = None
        return

    def reset_state(self) -> None:
        self.train_var = None
        return

    def __call__(self, outputs: Outputs, targets: Targets) -> dict[str, torch.Tensor]:
        ref_image = targets.ref
        SE = F.mse_loss(outputs.recon, ref_image, reduction='none')
        if self.train_var is None:
            self.train_var = torch.clamp(SE.mean(0).detach(), min=1e-5)

        local_device = outputs.recon.device
        train_var = torch.clamp(self.train_var.to(local_device), min=1e-5)
        train_logvar = torch.log(train_var)

        NLL = 0.5 * (SE / train_var + train_logvar).mean(0).sum() + self.adj_nll * train_var.numel()
        MSE = SE.mean()
        return {'NLL': NLL, 'MSE': MSE}


def get_nll_loss() -> LossBase[Outputs, Targets]:
    """Get Gaussian negative log-likelihood reconstruction loss component with MSE metric."""
    return DictLoss(NLLLoss(), name='NLL')


def _compute_prior_mu(
    q_mu: torch.Tensor,
    outputs: Outputs,
    _: Targets,
    dim_top_channel: int,
    scale_prior: float,
) -> torch.Tensor:
    """Compute prior mean tensor based on targets/conditions."""
    prior_mu = torch.zeros_like(q_mu)
    cond = outputs.condition
    if cond.dim() == 1:
        cond = cond.unsqueeze(1)
    for c in range(cond.size(1)):
        prior_mu[:, c * dim_top_channel : (c + 1) * dim_top_channel] = cond[:, c].view(-1, 1) * scale_prior

    return prior_mu


class KLDLoss:
    """Callable class for KL divergence loss calculation with freebits and raw KLD metric."""

    def __init__(
        self,
        freebits: float,
        scale_prior: float,
        dim_top_channel: int,
    ) -> None:
        self.freebits = freebits
        self.scale_prior = scale_prior
        self.dim_top_channel = dim_top_channel
        return

    def __call__(self, outputs: Outputs, targets: Targets) -> dict[str, torch.Tensor]:
        q_mu = outputs.mu[0]
        q_logvar = torch.clamp(outputs.log_var[0], min=-10.0, max=10.0)
        prior_mu = _compute_prior_mu(q_mu, outputs, targets, self.dim_top_channel, self.scale_prior)
        kld_matrix = -1 - q_logvar + (q_mu - prior_mu) ** 2 + q_logvar.exp()
        kld_free_bits = F.softplus(kld_matrix - 2 * self.freebits) + 2 * self.freebits
        kld = 0.5 * kld_free_bits.mean(0).sum()
        kld_real = 0.5 * kld_matrix.mean(0).sum()
        if len(outputs.mu) > 1 and outputs.prior_log_var:
            zipped = zip(outputs.mu[1:], outputs.log_var[1:], outputs.prior_log_var, strict=False)
            for d_mu, d_logvar, p_logvar in zipped:
                d_logvar = torch.clamp(d_logvar, min=-10.0, max=10.0)
                p_logvar = torch.clamp(p_logvar, min=-5.0, max=10.0)
                stage_matrix = -1 - d_logvar + d_logvar.exp() + (d_mu**2) * torch.exp(-p_logvar)
                s_free_bits = F.softplus(stage_matrix - 2 * self.freebits) + 2 * self.freebits

                kld = kld + 0.5 * s_free_bits.mean(0).sum()
                kld_real = kld_real + 0.5 * stage_matrix.mean(0).sum()

        return {'KLD': kld, 'KLD_real': kld_real}


def get_kld_loss() -> LossBase[Outputs, Targets]:
    """Get KL divergence loss component with KLD_real metric."""
    cfg = Experiment.get_config()
    objective_cfg = cfg.autoencoder.objective
    vae_arch = cfg.autoencoder.model.architecture
    n_resolutions = vae_arch.n_resolutions
    dim_top_channel = (2 ** (6 - n_resolutions) + 1) ** 2
    kld_fn = KLDLoss(
        freebits=objective_cfg.freebits,
        scale_prior=objective_cfg.scale_prior,
        dim_top_channel=dim_top_channel,
    )
    return DictLoss(kld_fn, name='KLD')


def get_annealing() -> LossBase[Outputs, Targets]:
    """Get (reverse) annealing component matching SurfaceVAE."""
    cfg = Experiment.get_config()
    annealing_period = cfg.autoencoder.train.annealing_period

    def _annealing(outputs: Outputs, _: Targets) -> torch.Tensor:
        device = outputs.recon.device
        time_fraction = torch.tensor(outputs.model_epoch / annealing_period, device=device)
        time_fraction = torch.clamp(time_fraction, 0.0, 1.0)
        return 0.5 * (1.0 - torch.cos(time_fraction * math.pi))

    return Loss(_annealing, name='Annealing')


def get_vae_loss() -> LossBase[Outputs, Targets]:
    """Get VAE loss composed with drytorch natural syntax."""
    cfg = Experiment.get_config()
    c_KLD = cfg.autoencoder.objective.c_KLD
    nll = get_nll_loss()
    kld = get_kld_loss()
    if cfg.autoencoder.train.annealing_period > 0:
        return nll + c_KLD * kld * get_annealing()

    return nll + c_KLD * kld


def get_classifier_loss() -> LossBase:
    """Get main loss objective based on active experiment configuration."""
    loss = get_cross_entropy()
    loss.watch(get_accuracy())
    return loss
