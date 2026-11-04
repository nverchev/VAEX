"""Counterfactuals of a sample towards every class, for each relaxation of the posterior."""

import dataclasses

from collections.abc import Sequence

import torch

from src.data.structures import Inputs
from src.lib.models import ClassifierModel, VAEModel

__all__ = [
    'CounterfactualSweep',
    'get_counterfactual_sweep',
]


@dataclasses.dataclass(frozen=True)
class CounterfactualSweep:
    """A sample with its counterfactuals towards every class, and what the classifier says of each image.

    Attributes:
        label: the class of the sample.
        relaxations: the weights of the posterior, from one (reconstruction) towards zero.
        original: the sample.
        recon: its reconstruction.
        cut: its simple cut, the decoding with no posterior shift.
        probabilities: the probability of the label for the original, the reconstruction and the cut.
        counterfactuals: one row per class, with one image per relaxation.
        counterfactual_probabilities: the probability of the target class for each counterfactual.
    """

    label: int
    relaxations: Sequence[float]
    original: torch.Tensor
    recon: torch.Tensor
    cut: torch.Tensor
    probabilities: torch.Tensor
    counterfactuals: Sequence[torch.Tensor]
    counterfactual_probabilities: Sequence[torch.Tensor]


@torch.inference_mode()
def get_counterfactual_sweep(
    classifier: ClassifierModel,
    vae: VAEModel,
    image: torch.Tensor,
    /,
    *,
    label: int,
    relaxations: Sequence[float],
) -> CounterfactualSweep:
    """Turn a sample towards every class, for each relaxation, and classify every image.

    Args:
        classifier: the classifier whose probabilities condition VAEX.
        vae: the trained VAE.
        image: the sample, of shape ``(C, H, W)``.
        label: the class of the sample.
        relaxations: the weights of the posterior of the lower latents.

    Returns:
        The images and the probabilities the classifier assigns to them.
    """
    batch = image.unsqueeze(0)
    inputs = Inputs(image=batch, log_probs=classifier.log_probabilities(Inputs(image=batch)))
    recon = vae.reconstruct(inputs).recon
    cut = vae.reconstruct(inputs, relaxation=0.0).recon
    probabilities = classifier.log_probabilities(Inputs(image=torch.cat([batch, recon, cut]))).exp()[:, label]
    counterfactuals: list[torch.Tensor] = []
    counterfactual_probabilities: list[torch.Tensor] = []
    for target in range(vae.module.n_classes):
        row = torch.cat([vae.generate_counterfactual(batch, target, relaxation).recon for relaxation in relaxations])
        counterfactuals.append(row)
        counterfactual_probabilities.append(classifier.log_probabilities(Inputs(image=row)).exp()[:, target])

    return CounterfactualSweep(
        label=label,
        relaxations=relaxations,
        original=image,
        recon=recon[0],
        cut=cut[0],
        probabilities=probabilities,
        counterfactuals=counterfactuals,
        counterfactual_probabilities=counterfactual_probabilities,
    )
