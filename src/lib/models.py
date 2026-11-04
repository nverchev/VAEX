"""Models of the experiment."""

from typing import Any, override

import torch

from drytorch.core.exceptions import ModelNotFoundError
from drytorch.lib.models import EMAModel, Model

from src.config.experiment import Experiment
from src.data.structures import Inputs, Outputs
from src.module.autoencoders import BaseVAEX, get_autoencoder
from src.module.classifiers import BaseClassifier, get_classifier

__all__ = [
    'ClassifierModel',
    'VAEEMAModel',
    'VAEModel',
    'get_autoencoder_model',
    'get_classifier_model',
    'load_autoencoder_model',
    'load_trained_state',
]


class ClassifierModel(Model[Inputs, torch.Tensor]):
    """Model wrapping a classifier that maps inputs to logits.

    Attributes:
        module: the classifier.
    """

    module: BaseClassifier

    @torch.inference_mode()
    def log_probabilities(self, inputs: Inputs, /) -> torch.Tensor:
        """Get the log-probabilities of the classes, in the precision of the model.

        Args:
            inputs: the images to classify.

        Returns:
            The log-probabilities, in full precision.
        """
        self.module.eval()
        return torch.log_softmax(self(inputs).float(), dim=-1)

    @torch.inference_mode()
    def describe_predictions(self, images: torch.Tensor, /) -> list[str]:
        """Describe the top class of each image and its probability, as in ``predicted 7 (0.98)``.

        Args:
            images: the images to classify.

        Returns:
            One description per image.
        """
        class_names = Experiment.get_config().data.dataset.classes
        probabilities, predictions = self.log_probabilities(Inputs(image=images)).exp().max(dim=1)
        return [
            f'predicted {class_names[prediction]} ({probability:.2f})'
            for prediction, probability in zip(predictions.tolist(), probabilities.tolist(), strict=True)
        ]


class VAEModel(Model[Inputs, Outputs]):
    """Model wrapping the VAE.

    Attributes:
        module: the VAE.
    """

    module: BaseVAEX

    @property
    def inference_module(self) -> BaseVAEX:
        """The VAE that reconstructs and generates: the trained one."""
        return self.module

    @torch.inference_mode()
    def generate(self, condition: torch.Tensor, /) -> torch.Tensor:
        """Generate images under a condition, sampling every latent from its prior.

        Args:
            condition: the condition of each image.

        Returns:
            The images, in full precision.
        """
        module = self.inference_module.eval()
        with self._autocast():
            return module.generate(condition).float()

    @torch.inference_mode()
    def generate_counterfactual(self, image: torch.Tensor, target: int, relaxation: float) -> Outputs:
        """Generate counterfactuals of the images for a target class, in the precision of the model.

        Args:
            image: the images to turn towards the target.
            target: the target class.
            relaxation: weight of the posterior of the lower latents.

        Returns:
            The outputs, with the counterfactuals in full precision.
        """
        module = self.inference_module.eval()
        with self._autocast():
            outputs = module.generate_counterfactual(image, target, relaxation)

        outputs.recon = outputs.recon.float()
        return outputs

    @torch.inference_mode()
    def reconstruct(self, inputs: Inputs, /, relaxation: float = 1.0) -> Outputs:
        """Reconstruct the images, in the precision of the model.

        Args:
            inputs: the images and the log-probabilities of the classifier.
            relaxation: weight of the posterior of the lower latents; ``1`` reconstructs the
                images, ``0`` gives their simple cut.

        Returns:
            The outputs, with the reconstructions in full precision.
        """
        module = self.inference_module.eval()
        with self._autocast():
            outputs = module.reconstruct(inputs, relaxation=relaxation)

        outputs.recon = outputs.recon.float()
        return outputs

    def _autocast(self) -> torch.autocast:
        return torch.autocast(self.device.type, dtype=self.mixed_precision, enabled=self.mixed_precision is not None)


class VAEEMAModel(EMAModel[Inputs, Outputs], VAEModel):
    """Model wrapping the VAE, with an exponential moving average of its weights.

    The average reconstructs and generates, and runs the forward pass in inference mode.

    Attributes:
        module: the trained VAE.
        averaged_module: the VAE with the averaged weights.
    """

    module: BaseVAEX
    averaged_module: BaseVAEX

    @property
    @override
    def inference_module(self) -> BaseVAEX:
        """The VAE that reconstructs and generates: the one with the averaged weights."""
        return self.averaged_module


def get_autoencoder_model(*, average: bool, mixed_precision: torch.dtype | None = None) -> VAEModel:
    """Get the VAE of the experiment.

    Args:
        average: whether to keep an exponential moving average of the weights.
        mixed_precision: if not None, the autocast precision.

    Returns:
        The VAE, wrapped for training, evaluation and checkpointing.
    """
    cfg = Experiment.get_config()
    cfg_model = cfg.autoencoder.model
    if average:
        return VAEEMAModel(
            get_autoencoder(),
            decay=cfg_model.ema_decay,
            name=cfg_model.name,
            device=cfg.user.device,
            mixed_precision=mixed_precision,
        )

    return VAEModel(get_autoencoder(), name=cfg_model.name, device=cfg.user.device, mixed_precision=mixed_precision)


def load_autoencoder_model() -> VAEModel:
    """Load the trained VAE with the average of its weights.

    Returns:
        The trained VAE, at the epoch ``user.load`` sets.

    Raises:
        RuntimeError: if the experiment holds no trained VAE.
    """
    cfg = Experiment.get_config()
    model = get_autoencoder_model(average=True)
    load_trained_state(model, epoch=cfg.user.load or -1)
    return model


def get_classifier_model() -> ClassifierModel:
    """Get the classifier of the experiment.

    Returns:
        The classifier, wrapped for training, evaluation and checkpointing.
    """
    cfg = Experiment.get_config()
    return ClassifierModel(get_classifier(), name=cfg.classification.model.name, device=cfg.user.device)


def load_trained_state(model: Model[Any, Any], /, *, epoch: int = -1) -> None:
    """Load the weights of a model trained by an earlier phase of the experiment.

    Args:
        model: the model to load.
        epoch: the epoch to load. Default loads the last saved epoch.

    Raises:
        RuntimeError: if the experiment holds no trained model.
    """
    try:
        model.load_state(epoch)
    except ModelNotFoundError as error:
        cfg = Experiment.get_config()
        msg = (
            f'No trained {model.name} for the experiment {cfg.name} in {cfg.user.path.version_dir}: '
            'train it first with the same configuration and overrides.'
        )
        raise RuntimeError(msg) from error

    return
