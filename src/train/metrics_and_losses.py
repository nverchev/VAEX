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



def get_accuracy() -> Metric[torch.Tensor, Targets]:
    """Return top-1 per-sample accuracy metric."""

    def _accuracy(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:

        return (outputs.argmax(dim=-1) == targets.label).float()

    return Metric[torch.Tensor, Targets](_accuracy, name='Accuracy')


def get_cross_entropy() -> Loss[torch.Tensor, Targets]:
    """Return cross-entropy loss objective."""
    ce = nn.CrossEntropyLoss(reduction='none')

    def _cross_entropy(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
        return ce(outputs, targets.label)

    return Loss[torch.Tensor, Targets](_cross_entropy, name='CrossEntropy')


class NLLLoss:
    """Callable class for Gaussian negative log-likelihood reconstruction loss component."""

    def __init__(self, variance: float = 0.1) -> None:
        self.adj_nll: float = 0.5 * math.log(2 * math.pi)
        self.var: float = variance
        self.logvar: float = math.log(self.var)
        return

    def __call__(self, outputs: Outputs, targets: Targets) -> dict[str, torch.Tensor]:
        ref_image = targets.ref
        SE = F.mse_loss(outputs.recon, ref_image, reduction='none')
        NLL = 0.5 * (SE / self.var + self.logvar).mean(0).sum() + self.adj_nll * SE[0].numel()
        MSE = SE.mean()
        return {'NLL': NLL, 'MSE': MSE}


def get_nll_loss() -> DictLoss[Outputs, Targets]:
    """Get Gaussian negative log-likelihood reconstruction loss component with MSE metric."""
    cfg = Experiment.get_config()
    objective_cfg = cfg.autoencoder.objective
    return DictLoss[Outputs, Targets](NLLLoss(variance=objective_cfg.sigma_2), name='NLL')


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


def get_kld_loss() -> DictLoss[Outputs, Targets]:
    """Get KL divergence loss component (free bits floor per latent stage)."""
    cfg = Experiment.get_config()
    return DictLoss[Outputs, Targets](KLDLoss(freebits=cfg.autoencoder.objective.freebits), name='KLD_FB')


def get_annealing() -> Loss[Outputs, Targets]:
    """Get the cyclical annealing weight on the KL term.

    The weight ramps to one over each cycle and restarts, so the model periodically gets
    the reconstruction pressure back and can re-enter the latents it had begun to give up.
    Past the annealing period the weight stays at one for the rest of training.
    """
    cfg = Experiment.get_config()
    annealing_period = cfg.autoencoder.train.annealing_period
    cycles = cfg.autoencoder.train.annealing_cycles

    min_weight = 0.01

    def _annealing(outputs: Outputs, _: Targets) -> torch.Tensor:
        device = outputs.recon.device
        progress = torch.tensor(outputs.model_epoch / annealing_period, device=device)
        if progress >= 1.0:
            return torch.ones((), device=device)

        phase = torch.remainder(progress * cycles, 1.0)
        weight = 0.5 * (1.0 - torch.cos(phase * math.pi))
        return torch.clamp(weight, min=min_weight)

    return Loss[Outputs, Targets](_annealing, name='Annealing')


def get_adversarial_loss() -> LossBase[Outputs, Targets]:
    """Get adversarial loss to erase classifier info."""
    kld = torch.nn.KLDivLoss(reduction='none', log_target=True)
    log_softmax = torch.nn.LogSoftmax(dim=1)
    cfg = Experiment.get_config()
    warmup = cfg.autoencoder.train.adv_warmup_epochs

    def _adv_kld(out: Outputs, _: Targets) -> torch.Tensor:
        loss = torch.zeros((), device=out.logits.device)
        for adv_logits in out.adv_logits:
            loss = loss + kld(log_softmax(adv_logits), out.logits).sum(1)
            
        if warmup > 0:
            scale = min(1.0, out.model_epoch / warmup)
            loss = loss * scale
        return loss

    return Loss[Outputs, Targets](_adv_kld, name='Adv')

def get_vae_loss() -> LossBase[Outputs, Targets]:
    """Get VAE loss composed with drytorch natural syntax."""
    cfg = Experiment.get_config()
    nll = get_nll_loss()
    kld = get_kld_loss()
    if cfg.autoencoder.train.annealing_period > 0:
        loss = nll + kld * get_annealing()
    else:
        loss = nll + kld

    from src.config.options import Models
    if cfg.autoencoder.model.name == Models.VAEX:
        loss = loss + get_adversarial_loss()

    return loss


def get_classifier_loss() -> LossBase[torch.Tensor, Targets]:
    """Get main loss objective based on active experiment configuration."""
    loss = get_cross_entropy()
    loss.watch(get_accuracy())
    return loss

