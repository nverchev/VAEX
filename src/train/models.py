"""Subclasses for the Model and EMAModel classes."""

from typing import cast

import torch

from drytorch.core import protocols as p
from drytorch.lib.models import EMAModel, Model

from src.config.experiment import Experiment
from src.data.structures import Inputs, Outputs
from src.module.autoencoder import get_autoencoder, Abstract_VAE
from src.module.classifier import BaseClassifier, get_classifier
from drytorch.core import exceptions as dry_exc


ClassifierModel = Model[Inputs, torch.Tensor]
ClassifierEMAModel = EMAModel[Inputs, torch.Tensor]


class LogEpochMixin(p.ModelProtocol[Inputs, Outputs]):
    """Mixin that stamps the current training epoch into every forward-pass output."""

    def __call__(self, inputs: Inputs) -> Outputs:
        out = super().__call__(inputs)
        out.model_epoch = self.epoch
        return out


class VAEModel(LogEpochMixin, Model[Inputs, Outputs]):
    """VAE model with epoch logging."""


class VAEEMAModel(LogEpochMixin, EMAModel[Inputs, Outputs]):
    """VAE EMA model with epoch logging."""


def load_extract_autoencoder_module() -> Abstract_VAE:
    """Load and extract the autoencoder module from the EMA model."""
    cfg = Experiment.get_config()
    autoencoder = get_autoencoder().eval()
    model = VAEEMAModel(autoencoder, name=cfg.autoencoder.name, device=cfg.user.device)
    epoch = cfg.user.load if cfg.user.load != 0 else -1
    model.load_state(epoch)
    module = cast(Abstract_VAE, model.averaged_module)
    module.eval()
    return module


def load_extract_classifier_module() -> BaseClassifier:
    """Load and extract the classifier module from the latest checkpoint."""
    cfg = Experiment.get_config()
    classifier = get_classifier()
    model = ClassifierModel(classifier, name=cfg.classification.name, device=cfg.user.device)
    try:
        model.load_state(-1)
    except (dry_exc.ModelNotFoundError, dry_exc.EpochNotFoundError) as err:
        msg = f"VAEX requires a pre-trained classifier in the same experiment '{cfg.name}'."
        raise RuntimeError(msg) from err

    classifier.eval()
    return classifier


__all__ = [
    'ClassifierEMAModel',
    'ClassifierModel',
    'VAEEMAModel',
    'VAEModel',
    'load_extract_autoencoder_module',
    'load_extract_classifier_module',
]

