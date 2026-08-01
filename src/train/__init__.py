"""Subpackage for training, evaluation and losses."""

from src.train.learning_schema import get_learning_schema
from src.train.loaders import get_loaders
from src.train.metrics_and_losses import get_classifier_loss, get_accuracy


__all__ = [
    'get_accuracy',
    'get_learning_schema',
    'get_loaders',
    'get_classifier_loss',
]
