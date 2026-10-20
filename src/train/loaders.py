"""DataLoader factory for training and evaluation."""

from typing import TYPE_CHECKING

from drytorch import DataLoader
from torch.utils.data import Dataset

from src.data.dataset import get_splits
from src.data.derived_datasets import EvaluatedDataset
from src.data.structures import Inputs, Targets

if TYPE_CHECKING:
    from src.train.models import ClassifierModel


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
