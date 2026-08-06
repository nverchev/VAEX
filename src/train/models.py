"""Subclasses for the Model and EMAModel classes."""

import logging

from typing import cast

import torch

from drytorch.core import protocols as p
from drytorch.lib.models import EMAModel, Model

from src.config.experiment import Experiment
from src.data.structures import Inputs, Outputs
from src.module.autoencoder import get_autoencoder, Abstract_VAE
from src.module.classifier import BaseClassifier, get_classifier
from drytorch.core import exceptions as dry_exc

GENERATIVE_SUFFIX = '_bn'

ClassifierModel = Model[Inputs, torch.Tensor]


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


class GenerativeVAEModel(VAEModel):
    """VAE model that decodes with the decaying posterior weights of its module.

    ``Abstract_VAE.forward`` always keeps the whole posterior, which is what training
    needs. Generation and counterfactuals instead decode with ``posterior_weights``, and
    running that path through the model rather than through the caller lets any drytorch
    runner drive it.
    """

    def __call__(self, inputs: Inputs) -> Outputs:
        """Encode, then decode with the generative posterior weights."""
        module = cast(Abstract_VAE, self.module)
        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            condition = module.condition_from(inputs)
            out = module.decode(module.encode(inputs.image), condition, module.posterior_weights)

        out.image = inputs.image
        out.condition = condition
        out.model_epoch = self.epoch
        return out


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


def clone_for_generation(model: VAEEMAModel) -> GenerativeVAEModel:
    """Copy the module used at inference into a model of its own.

    The weights move through a state dictionary into a freshly built module, which is
    both lighter than a deep copy and mirrors how the checkpoint is read back. It also
    gives the clone a distinct module, as drytorch keys its registry on module identity
    and would refuse to register the same object twice.

    Args:
        model: the trained model whose averaged module is copied.

    Returns:
        A model of its own at the same epoch, safe to modify and save.
    """
    cfg = Experiment.get_config()
    clone = GenerativeVAEModel(
        get_autoencoder().eval(),
        name=f'{cfg.autoencoder.name}{GENERATIVE_SUFFIX}',
        device=model.device,
        should_compile=False,
        should_distribute=False,
    )
    clone.module.load_state_dict(model.averaged_module.state_dict())
    clone.epoch = model.epoch
    return clone


def load_extract_generative_module() -> Abstract_VAE:
    """Load the autoencoder whose batch normalization suits the generative path.

    Falls back to the plain averaged module when no such model was saved, so runs that
    predate it, or that disabled ``recalibrate_bn``, still load.
    """
    cfg = Experiment.get_config()
    autoencoder = get_autoencoder().eval()
    model = VAEModel(
        autoencoder,
        name=f'{cfg.autoencoder.name}{GENERATIVE_SUFFIX}',
        device=cfg.user.device,
    )
    epoch = cfg.user.load if cfg.user.load != 0 else -1
    try:
        model.load_state(epoch)
    except (dry_exc.ModelNotFoundError, dry_exc.EpochNotFoundError):
        logging.warning('No batch-norm recalibrated model found; using the averaged one.')
        return load_extract_autoencoder_module()

    module = cast(Abstract_VAE, model.module)
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
    'GENERATIVE_SUFFIX',
    'ClassifierModel',
    'GenerativeVAEModel',
    'VAEEMAModel',
    'VAEModel',
    'clone_for_generation',
    'load_extract_autoencoder_module',
    'load_extract_classifier_module',
    'load_extract_generative_module',
]


