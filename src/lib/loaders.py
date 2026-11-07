"""DataLoader factory for training and evaluation."""

from typing import TYPE_CHECKING

from drytorch import DataLoader
from torch.utils.data import Dataset

from src.config.experiment import Experiment
from src.data.dataset import get_splits
from src.data.derived_datasets import EvaluatedDataset
from src.data.split import Subset
from src.data.structures import Inputs, Targets

if TYPE_CHECKING:
    from src.lib.models import ClassifierModel


def get_loaders(
    batch_size: int,
    n_workers: int = 0,
    classifier: 'ClassifierModel | None' = None,
) -> tuple[DataLoader[tuple[Inputs, Targets]], DataLoader[tuple[Inputs, Targets]]]:
    """Return (train_loader, eval_loader) for the current experiment phase."""
    train_split, eval_split = get_splits()
    # the loaders read the splits, or the splits labelled by the classifier
    train_dataset: Dataset[tuple[Inputs, Targets]] = train_split
    eval_dataset: Dataset[tuple[Inputs, Targets]] = eval_split
    if classifier is not None:
        train_dataset = EvaluatedDataset(train_split, classifier)
        eval_dataset = EvaluatedDataset(eval_split, classifier)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, n_workers=0, pin_memory=False)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, n_workers=n_workers, pin_memory=False)

    return train_loader, eval_loader


def get_calibration_loader(classifier: 'ClassifierModel | None', /) -> DataLoader[tuple[Inputs, Targets]]:
    """Return the first ``evaluation.bn_samples`` training images, to re-estimate the batch normalization of the VAE.

    Args:
        classifier: the trained classifier labelling the images, which the reconstructions and the
            counterfactuals need; ``None`` suffices for the generation, which reads no image.

    Returns:
        The loader of the images, with the log-probabilities of the classifier when given.
    """
    cfg = Experiment.get_config()
    train_split, _ = get_splits()
    subset = Subset(train_split, range(cfg.evaluation.bn_samples))
    dataset: Dataset[tuple[Inputs, Targets]] = subset if classifier is None else EvaluatedDataset(subset, classifier)
    return DataLoader(dataset=dataset, batch_size=cfg.autoencoder.train.batch_size, pin_memory=False)
