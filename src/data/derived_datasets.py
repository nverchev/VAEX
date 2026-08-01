"""Derived datasets for classification evaluation, autoencoder reconstruction, and counterfactual generation."""

import abc
from collections.abc import Sized
from typing import Any, Generic, TypeVar, override

import torch
from torch.utils.data import Dataset

from src.data.structures import Inputs, Outputs, Targets
from src.module.autoencoder import Abstract_VAE
from src.module.classifier import BaseClassifier

V = TypeVar('V', bound=Abstract_VAE)


class ClassifierMixin:
    """Mixin for models with classifier inference."""

    classifier: BaseClassifier

    @property
    def device(self) -> torch.device:
        return next(self.classifier.parameters()).device

    @torch.inference_mode()
    def _classify(self, batch_inputs: Inputs) -> torch.Tensor:
        self.classifier.eval()
        return self.classifier(batch_inputs)


class AutoencoderMixin(Generic[V]):
    """Mixin for autoencoder inference and reconstruction."""

    autoencoder: V

    @property
    def device(self) -> torch.device:
        return next(self.autoencoder.parameters()).device

    @torch.inference_mode()
    def _reconstruct(self, batch_inputs: Inputs) -> Outputs:
        self.autoencoder.eval()
        return self.autoencoder(batch_inputs)


class CounterfactualMixin(ClassifierMixin, AutoencoderMixin[Abstract_VAE]):
    """Mixin for generating counterfactual reconstructions."""

    target_class: int

    @property
    def device(self) -> torch.device:
        classifier_device = next(self.classifier.parameters()).device
        autoencoder_device = next(self.autoencoder.parameters()).device
        if classifier_device != autoencoder_device:
            raise ValueError('Autoencoder and classifier must be on the same device')

        return autoencoder_device

    @torch.inference_mode()
    def _generate_counterfactual(self, batch_inputs: Inputs) -> Outputs:
        self.autoencoder.eval()
        return self.autoencoder.generate_counterfactual(batch_inputs, target_dim=self.target_class)


class ProcessedDataset(abc.ABC):
    """Base derived dataset with common collation and device transfer functionality."""

    def __init__(self, dataset: Dataset[tuple[Inputs, Targets]], pin_memory: bool = False) -> None:
        self.dataset = dataset
        self.pin_memory = pin_memory
        return

    def __len__(self) -> int:
        assert isinstance(self.dataset, Sized)
        return len(self.dataset)

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        pass

    def _collate(self, index_list: list[int]) -> tuple[Inputs, Targets]:
        batch_data = [self.dataset[i] for i in index_list]
        image = torch.stack([d[0].image for d in batch_data])
        labels = torch.stack([d[1].label for d in batch_data])
        refs = torch.stack([d[1].ref if d[1].ref.numel() > 0 else d[0].image for d in batch_data])
        if self.pin_memory:
            image, labels, refs = image.pin_memory(), labels.pin_memory(), refs.pin_memory()

        image = image.to(self.device, non_blocking=self.pin_memory)
        labels = labels.to(self.device, non_blocking=self.pin_memory)
        refs = refs.to(self.device, non_blocking=self.pin_memory)
        return Inputs(image=image), Targets(ref=refs, label=labels)

    @staticmethod
    def _unbind(batch_inputs: Inputs, batch_targets: Targets) -> list[tuple[Inputs, Targets]]:
        batch_size = len(batch_targets.label)
        has_logits = batch_inputs.logits.numel() > 0
        return [
            (
                Inputs(image=batch_inputs.image[i], logits=batch_inputs.logits[i] if has_logits else torch.empty(0)),
                Targets(ref=batch_targets.ref[i], label=batch_targets.label[i]),
            )
            for i in range(batch_size)
        ]

    @abc.abstractmethod
    def __getitems__(self, index_list: list[int]) -> list[Any]:
        """Get pre-processed data for the given indices."""
        pass


class EvaluatedDataset(ClassifierMixin, ProcessedDataset, Dataset[tuple[Inputs, Targets]]):
    """Dataset that attaches classifier logits to each sample."""

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        classifier: BaseClassifier,
    ) -> None:
        super().__init__(dataset=dataset)
        self.classifier = classifier
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        batch_inputs, batch_targets = self._collate(index_list)
        batch_inputs = batch_inputs._replace(logits=self._classify(batch_inputs))
        return self._unbind(batch_inputs, batch_targets)


class ReconstructedDataset(AutoencoderMixin[V], ProcessedDataset, Generic[V], Dataset[tuple[Inputs, Targets]]):
    """Dataset that replaces input features with autoencoder reconstructions."""

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: V,
    ) -> None:
        super().__init__(dataset=dataset)
        self.autoencoder = autoencoder
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        batch_inputs, batch_targets = self._collate(index_list)
        recons = self._reconstruct(batch_inputs).recon
        if recons is None:
            recons = batch_inputs.image

        batch_inputs = Inputs(image=recons)
        batch_targets = Targets(ref=recons, label=batch_targets.label)
        return self._unbind(batch_inputs, batch_targets)


class ReconstructedEvaluatedDataset(CounterfactualMixin, ProcessedDataset, Dataset[tuple[Inputs, Targets]]):
    """Dataset that classifies inputs first, then reconstructs using logits."""

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: Abstract_VAE,
        classifier: BaseClassifier,
    ) -> None:
        super().__init__(dataset=dataset)
        self.autoencoder = autoencoder
        self.classifier = classifier
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        batch_inputs, batch_targets = self._collate(index_list)
        batch_inputs = batch_inputs._replace(logits=self._classify(batch_inputs))
        recons = self._reconstruct(batch_inputs).recon
        if recons is None:
            recons = batch_inputs.image

        batch_inputs = Inputs(image=recons, logits=batch_inputs.logits)
        batch_targets = Targets(ref=recons, label=batch_targets.label)
        return self._unbind(batch_inputs, batch_targets)


class CounterfactualDataset(CounterfactualMixin, ProcessedDataset, Dataset[tuple[Inputs, Targets]]):
    """Dataset that generates counterfactual reconstructions conditioned on a target class."""

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: Abstract_VAE,
        classifier: BaseClassifier,
        target_class: int = 1,
    ) -> None:
        super().__init__(dataset=dataset)
        self.autoencoder = autoencoder
        self.classifier = classifier
        self.target_class = target_class
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        batch_inputs, batch_targets = self._collate(index_list)
        batch_inputs = batch_inputs._replace(logits=self._classify(batch_inputs))
        recons = self._generate_counterfactual(batch_inputs).recon
        if recons is None:
            recons = batch_inputs.image

        batch_inputs = Inputs(image=recons, logits=batch_inputs.logits)
        target_tensor = torch.full_like(batch_targets.label, self.target_class)
        batch_targets = Targets(ref=recons, label=target_tensor)
        return self._unbind(batch_inputs, batch_targets)
