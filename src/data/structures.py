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
        mu: Shift of the posterior mean of the top latent from its prior, flattened.
        log_var: Shift of the posterior log-variance of the top latent from its prior, flattened.
        hidden: Features of the encoder at each resolution, from the coarsest.
    """

    mu: torch.Tensor
    log_var: torch.Tensor
    hidden: tuple[torch.Tensor, ...]


def _get_empty() -> torch.Tensor:
    return torch.empty(0)


@dataclasses.dataclass(slots=True)
class Outputs:
    """Outputs of the VAE.

    Attributes:
        recon: Reconstructed (or counterfactual) images.
        mu: Shifts of the posterior means from the priors, from the top latent down.
        log_var: Shifts of the posterior log-variances from the priors, from the top latent down.
        prior_log_var: Prior log-variances, from the top latent down.
        adv_logits: Logits of the adversary on the free channels of the top latent.
        log_probs: Log-probabilities of the classifier, which the adversary learns to predict.
        epoch: Epoch of the model when it produced the outputs, 1 in the first training epoch.
    """

    recon: torch.Tensor
    mu: list[torch.Tensor] = dataclasses.field(default_factory=list)
    log_var: list[torch.Tensor] = dataclasses.field(default_factory=list)
    prior_log_var: list[torch.Tensor] = dataclasses.field(default_factory=list)
    adv_logits: torch.Tensor = dataclasses.field(default_factory=_get_empty)
    log_probs: torch.Tensor = dataclasses.field(default_factory=_get_empty)
    epoch: int = 0
