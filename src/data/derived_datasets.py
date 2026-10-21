"""Derived datasets for classification evaluation, autoencoder reconstruction, and counterfactual generation."""

from __future__ import annotations

import abc
from collections.abc import Sized
from typing import TYPE_CHECKING, Any, override

import torch

from src.data.split import DataSplit
from src.data.structures import Inputs, Outputs, Targets

if TYPE_CHECKING:
    from src.lib.models import ClassifierModel, VAEModel


class ClassifierMixin:
    """Mixin for models with classifier inference."""

    classifier: ClassifierModel

    @property
    def device(self) -> torch.device:
        return self.classifier.device

    def _classify(self, batch_inputs: Inputs) -> torch.Tensor:
        """Return the classifier's log-probabilities."""
        return self.classifier.log_probabilities(batch_inputs)


class AutoencoderMixin:
    """Mixin for autoencoder inference and reconstruction."""

    autoencoder: VAEModel

    @property
    def device(self) -> torch.device:
        return self.autoencoder.device

    def _reconstruct(self, batch_inputs: Inputs) -> Outputs:
        return self.autoencoder.reconstruct(batch_inputs)


class CounterfactualMixin(ClassifierMixin, AutoencoderMixin):
    """Mixin for generating counterfactual reconstructions."""

    target_class: int
    relaxation: float

    @property
    def device(self) -> torch.device:
        if self.classifier.device != self.autoencoder.device:
            raise ValueError('Autoencoder and classifier must be on the same device')

        return self.autoencoder.device

    def _generate_counterfactual(self, batch_inputs: Inputs) -> Outputs:
        return self.autoencoder.generate_counterfactual(batch_inputs.image, self.target_class, self.relaxation)


class ProcessedDataset(DataSplit, abc.ABC):
    """Base derived dataset with common collation and device transfer functionality."""

    split: DataSplit
    pin_memory: bool

    def __init__(self, split: DataSplit, pin_memory: bool = False) -> None:
        super().__init__()
        self.split = split
        self.pin_memory = pin_memory
        return

    def __len__(self) -> int:
        assert isinstance(self.split, Sized)
        return len(self.split)

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        return self.__getitems__([index])[0]

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """Compute device for batch data transfer."""

    def _collate(self, index_list: list[int]) -> tuple[Inputs, Targets]:
        batch_data = [self.split[i] for i in index_list]
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
        has_log_probs = batch_inputs.log_probs.numel() > 0
        empty = torch.empty(0)
        return [
            (
                Inputs(image=batch_inputs.image[i], log_probs=batch_inputs.log_probs[i] if has_log_probs else empty),
                Targets(ref=batch_targets.ref[i], label=batch_targets.label[i]),
            )
            for i in range(batch_size)
        ]

    @abc.abstractmethod
    def __getitems__(self, index_list: list[int]) -> list[Any]:
        """Get pre-processed data for the given indices."""


class EvaluatedDataset(ClassifierMixin, ProcessedDataset):
    """Dataset that attaches classifier log-probabilities to each sample."""

    def __init__(
        self,
        split: DataSplit,
        classifier: ClassifierModel,
    ) -> None:
        super().__init__(split=split)
        self.classifier = classifier
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        batch_inputs, batch_targets = self._collate(index_list)
        batch_inputs = batch_inputs._replace(log_probs=self._classify(batch_inputs))
        return self._unbind(batch_inputs, batch_targets)


class ReconstructedDataset(AutoencoderMixin, ProcessedDataset):
    """Dataset that replaces input features with autoencoder reconstructions."""

    def __init__(
        self,
        split: DataSplit,
        autoencoder: VAEModel,
    ) -> None:
        super().__init__(split=split)
        self.autoencoder = autoencoder
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        batch_inputs, batch_targets = self._collate(index_list)
        recons = self._reconstruct(batch_inputs).recon
        batch_inputs = Inputs(image=recons)
        batch_targets = Targets(ref=recons, label=batch_targets.label)
        return self._unbind(batch_inputs, batch_targets)


class ReconstructedEvaluatedDataset(CounterfactualMixin, ProcessedDataset):
    """Dataset that classifies inputs first, then reconstructs using the classifier distribution."""

    def __init__(
        self,
        split: DataSplit,
        autoencoder: VAEModel,
        classifier: ClassifierModel,
    ) -> None:
        super().__init__(split=split)
        self.autoencoder = autoencoder
        self.classifier = classifier
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        batch_inputs, batch_targets = self._collate(index_list)
        batch_inputs = batch_inputs._replace(log_probs=self._classify(batch_inputs))
        recons = self._reconstruct(batch_inputs).recon
        batch_inputs = Inputs(image=recons, log_probs=batch_inputs.log_probs)
        batch_targets = Targets(ref=recons, label=batch_targets.label)
        return self._unbind(batch_inputs, batch_targets)


class CounterfactualDataset(CounterfactualMixin, ProcessedDataset):
    """Dataset that generates counterfactual reconstructions conditioned on a target class."""

    def __init__(
        self,
        split: DataSplit,
        autoencoder: VAEModel,
        classifier: ClassifierModel,
        target_class: int = 1,
        relaxation: float = 1.0,
    ) -> None:
        super().__init__(split=split)
        self.autoencoder = autoencoder
        self.classifier = classifier
        self.target_class = target_class
        self.relaxation = relaxation
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        batch_inputs, batch_targets = self._collate(index_list)
        batch_inputs = batch_inputs._replace(log_probs=self._classify(batch_inputs))
        recons = self._generate_counterfactual(batch_inputs).recon
        batch_inputs = Inputs(image=recons, log_probs=batch_inputs.log_probs)
        target_tensor = torch.full_like(batch_targets.label, self.target_class)
        batch_targets = Targets(ref=recons, label=target_tensor)
        return self._unbind(batch_inputs, batch_targets)
