"""DataLoader factory for training and evaluation."""

from typing import TYPE_CHECKING

from drytorch import DataLoader

from src.data.dataset import get_datasets
from src.data.derived_datasets import EvaluatedDataset

if TYPE_CHECKING:
    from src.module.classifier import BaseClassifier

def get_loaders(
    batch_size: int,
    n_workers: int = 0,
    classifier: 'BaseClassifier | None' = None,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, eval_loader) for the current experiment phase."""
    train_dataset, eval_dataset = get_datasets()
    if classifier is not None:
        train_dataset = EvaluatedDataset(train_dataset, classifier)
        eval_dataset = EvaluatedDataset(eval_dataset, classifier)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, n_workers=0, pin_memory=False)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, n_workers=n_workers, pin_memory=False)

    return train_loader, eval_loader
