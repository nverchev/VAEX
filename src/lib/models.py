"""Models of the experiment."""

from collections.abc import Callable, Iterable, Iterator
from typing import Any, Final, override

import torch
import torch.nn as nn

from drytorch.core.exceptions import ModelNotFoundError
from drytorch.lib.models import EMAModel, Model

from src.config.experiment import Experiment
from src.data.structures import Inputs, Outputs, Targets
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


type BatchNormPath = tuple[str, float]
"""A path through the VAE: the method and the relaxation."""

BATCH_NORM_TYPES: Final = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


class VAEModel(Model[Inputs, Outputs]):
    """Model wrapping the VAE.

    In evaluation mode, the latents are sampled with a smaller noise, and the counterfactuals
    and the generation decode along other paths than training, so the batch normalization
    statistics that training collected do not fit them. With calibration inputs, the model
    re-estimates the statistics of the module it evaluates on each path before it first takes
    it, and keeps them for each path. The counterfactuals of one relaxation share statistics,
    measured with each image turned to another class than its prediction.

    Attributes:
        module: the VAE.
        calibration: batches of training images, labelled by the classifier, on which the batch
            normalization statistics are re-estimated; ``None`` keeps those of training.
    """

    module: BaseVAEX
    calibration: Iterable[tuple[Inputs, Targets]] | None = None
    _batch_norm_path: BatchNormPath | None = None
    _batch_norm_cache: dict[BatchNormPath, list[torch.Tensor]] | None = None

    def set_calibration(self, loader: Iterable[tuple[Inputs, Targets]], /) -> None:
        """Re-estimate the batch normalization statistics on these batches for each path.

        Args:
            loader: batches of training images with the log-probabilities of the classifier, on
                any device; see :func:`src.lib.loaders.get_calibration_loader`.
        """
        self.calibration = loader
        self._batch_norm_path = None
        self._batch_norm_cache = {}
        return

    @override
    def __call__(self, inputs: Inputs, /) -> Outputs:
        """Run the forward pass, recording the epoch of the model for the annealing and the warm-up."""
        if torch.is_inference_mode_enabled():
            self._calibrate(('reconstruct', 1.0), self._run_reconstruction)

        outputs = super().__call__(inputs)
        outputs.epoch = self.epoch
        return outputs

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
        self._calibrate(('generate', 0.0), self._run_generation)
        module = self.inference_module.eval()
        with self._autocast():
            return module.generate(condition).float()

    @torch.inference_mode()
    def generate_counterfactual(self, inputs: Inputs, target: int, relaxation: float) -> Outputs:
        """Generate counterfactuals of the images for a target class, in the precision of the model.

        Args:
            inputs: the images to turn towards the target and the log-probabilities of the classifier.
            target: the target class.
            relaxation: weight of the posterior of the lower latents.

        Returns:
            The outputs, with the counterfactuals in full precision.
        """
        self._calibrate(('counterfactual', relaxation), self._run_counterfactual)
        module = self.inference_module.eval()
        with self._autocast():
            outputs = module.generate_counterfactual(inputs, target, relaxation)

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
        self._calibrate(('reconstruct', relaxation), self._run_reconstruction)
        module = self.inference_module.eval()
        with self._autocast():
            outputs = module.reconstruct(inputs, relaxation=relaxation)

        outputs.recon = outputs.recon.float()
        return outputs

    def _autocast(self) -> torch.autocast:
        return torch.autocast(self.device.type, dtype=self.mixed_precision, enabled=self.mixed_precision is not None)

    def _calibrate(self, path: BatchNormPath, run: Callable[[BaseVAEX, Inputs, BatchNormPath], object]) -> None:
        if self.calibration is None or self._batch_norm_cache is None or path == self._batch_norm_path:
            return

        module = self.inference_module.eval()
        batch_norms = [layer for layer in module.modules() if isinstance(layer, BATCH_NORM_TYPES)]
        if path in self._batch_norm_cache:
            for layer, cached in zip(batch_norms, _pairs(self._batch_norm_cache[path]), strict=True):
                for buffer, value in zip(_get_statistics(layer), cached, strict=True):
                    buffer.copy_(value)

            self._batch_norm_path = path
            return

        momenta = [layer.momentum for layer in batch_norms]
        for layer in batch_norms:
            layer.reset_running_stats()
            layer.momentum = None  # a cumulative average over the calibration inputs
            layer.train()

        try:
            with self._autocast():
                for inputs, _ in self.calibration:
                    batch = Inputs(image=inputs.image.to(self.device), log_probs=inputs.log_probs.to(self.device))
                    run(module, batch, path)
        finally:
            for layer, momentum in zip(batch_norms, momenta, strict=True):
                layer.momentum = momentum
                layer.eval()

        statistics: list[torch.Tensor] = []
        for layer in batch_norms:
            statistics.extend(buffer.clone() for buffer in _get_statistics(layer))

        self._batch_norm_cache[path] = statistics
        self._batch_norm_path = path
        return

    @staticmethod
    def _run_reconstruction(module: BaseVAEX, batch: Inputs, path: BatchNormPath) -> None:
        module.reconstruct(batch, relaxation=path[1])
        return

    @staticmethod
    def _run_counterfactual(module: BaseVAEX, batch: Inputs, path: BatchNormPath) -> None:
        # each image turns to the next class from its prediction, so that the population mixes the
        # classes as in training, and the shift towards a class is not normalised away
        targets = (batch.log_probs.argmax(dim=1) + 1) % module.n_classes
        module.generate_counterfactual(batch, targets, path[1])
        return

    @staticmethod
    def _run_generation(module: BaseVAEX, batch: Inputs, _: BatchNormPath) -> None:
        module.generate(module.get_random_condition(len(batch.image)))
        return


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


def _pairs(statistics: list[torch.Tensor]) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    return zip(statistics[::2], statistics[1::2], strict=True)


def _get_statistics(layer: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    mean, var = layer.running_mean, layer.running_var
    if not isinstance(mean, torch.Tensor) or not isinstance(var, torch.Tensor):
        raise ValueError(f'{layer} does not track running statistics to re-estimate.')

    return mean, var
