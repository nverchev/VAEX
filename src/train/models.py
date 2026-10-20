"""Models of the experiment."""

from typing import Any

import torch

from drytorch.core.exceptions import ModelNotFoundError
from drytorch.lib.models import Model

from src.config.experiment import Experiment
from src.data.structures import Inputs, Outputs
from src.module.autoencoder import BaseVAEX, get_autoencoder
from src.module.classifier import BaseClassifier, get_classifier

__all__ = [
    'ClassifierModel',
    'VAEModel',
    'get_autoencoder_model',
    'get_classifier_model',
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
        self.module.eval()
        with self._autocast():
            outputs = self.module.generate_counterfactual(image, target, relaxation)

        outputs.recon = outputs.recon.float()
        return outputs

    @torch.inference_mode()
    def reconstruct(self, inputs: Inputs, /) -> Outputs:
        """Reconstruct the images, in the precision of the model.

        Args:
            inputs: the images and the log-probabilities of the classifier.

        Returns:
            The outputs, with the reconstructions in full precision.
        """
        self.module.eval()
        with self._autocast():
            outputs = self.module.reconstruct(inputs)

        outputs.recon = outputs.recon.float()
        return outputs

    def _autocast(self) -> torch.autocast:
        return torch.autocast(self.device.type, dtype=self.mixed_precision, enabled=self.mixed_precision is not None)


def get_autoencoder_model(*, mixed_precision: torch.dtype | None = None) -> VAEModel:
    """Get the VAE of the experiment.

    Args:
        mixed_precision: if not None, the autocast precision.

    Returns:
        The VAE, wrapped for training, evaluation and checkpointing.
    """
    cfg = Experiment.get_config()
    name = cfg.autoencoder.name
    return VAEModel(get_autoencoder(), name=name, device=cfg.user.device, mixed_precision=mixed_precision)


def get_classifier_model() -> ClassifierModel:
    """Get the classifier of the experiment.

    Returns:
        The classifier, wrapped for training, evaluation and checkpointing.
    """
    cfg = Experiment.get_config()
    return ClassifierModel(get_classifier(), name=cfg.classification.name, device=cfg.user.device)


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
