"""Typed I/O structures for the model, loss and metrics."""

import dataclasses
from typing import NamedTuple

import torch


class Inputs(NamedTuple):
    """Input batch containing image tensor, condition tensor, and classifier log-probabilities."""

    image: torch.Tensor
    condition: torch.Tensor = torch.empty(0)
    log_probs: torch.Tensor = torch.empty(0)

    @property
    def features(self) -> torch.Tensor:
        """Alias for image compatibility with generic linear layers."""
        return self.image


class Targets(NamedTuple):
    """Ground-truth target labels and reference image for loss functions and metrics."""

    label: torch.Tensor
    ref: torch.Tensor = torch.empty(0)


@dataclasses.dataclass(frozen=True)
class Encoding:
    """Encoder outputs that the decoder reads.

    Attributes:
        z: Sample of the top latent, flattened.
        mu: Mean of the posterior of the top latent.
        log_var: Log-variance of the posterior of the top latent.
        hidden: Features of the encoder at each resolution, from the coarsest.
    """

    z: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
    hidden: tuple[torch.Tensor, ...]


@dataclasses.dataclass(slots=True)
class Outputs:
    """Outputs of the VAE.

    Attributes:
        recon: Reconstructed (or counterfactual) images.
        prior_mu: Prior mean of the top latent, set by the condition.
        mu: Mean of the posterior of the top latent, then the posterior shifts of the stages.
        log_var: Log-variance of the posterior of the top latent, then the shifts of the stages.
        prior_log_var: Prior log-variances of the stages.
    """

    recon: torch.Tensor
    prior_mu: torch.Tensor
    mu: list[torch.Tensor] = dataclasses.field(default_factory=list)
    log_var: list[torch.Tensor] = dataclasses.field(default_factory=list)
    prior_log_var: list[torch.Tensor] = dataclasses.field(default_factory=list)
