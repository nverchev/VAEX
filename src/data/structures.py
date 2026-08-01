"""Typed I/O structures for the model, loss and metrics."""

import dataclasses
from typing import NamedTuple, Self

import torch


class Inputs(NamedTuple):
    """Input batch containing image tensor, condition tensor, and logits."""

    image: torch.Tensor
    condition: torch.Tensor = torch.empty(0)
    logits: torch.Tensor = torch.empty(0)

    @property
    def features(self) -> torch.Tensor:
        """Alias for image compatibility with generic linear layers."""
        return self.image


class Targets(NamedTuple):
    """Ground-truth target labels and reference image for loss functions and metrics."""

    label: torch.Tensor
    ref: torch.Tensor = torch.empty(0)


@dataclasses.dataclass(slots=True)
class Outputs:
    """Outputs produced by model forward passes containing reconstructions, latent statistics, and logits."""

    recon: torch.Tensor = dataclasses.field(init=False)
    image: torch.Tensor = dataclasses.field(init=False)
    logits: torch.Tensor = dataclasses.field(init=False)

    mu: list[torch.Tensor] = dataclasses.field(default_factory=list)
    log_var: list[torch.Tensor] = dataclasses.field(default_factory=list)
    prior_mu: list[torch.Tensor] = dataclasses.field(default_factory=list)
    prior_log_var: list[torch.Tensor] = dataclasses.field(default_factory=list)
    condition: torch.Tensor = dataclasses.field(init=False)
    hidden: list[torch.Tensor] = dataclasses.field(default_factory=list)
    z: list[torch.Tensor] = dataclasses.field(default_factory=list)
    model_epoch: int = 0

    def update(self, other: Self) -> None:
        """Update output fields from another outputs instance."""
        for attribute in self.__slots__:
            if hasattr(other, attribute):
                setattr(self, attribute, getattr(other, attribute))

        return None
