"""Tests for the losses and metrics."""

import torch

from src.data.structures import Targets
from src.train.objectives import get_classifier_loss


def test_the_classifier_loss_reports_the_accuracy() -> None:
    """The cross entropy is joined with the accuracy, which is reported but not optimized."""
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    targets = Targets(label=torch.tensor([0, 1, 1]))

    result = get_classifier_loss().calculate(logits, targets)

    assert set(result) == {'CrossEntropy', 'Accuracy'}
    torch.testing.assert_close(result['Accuracy'], torch.tensor([1.0, 1.0, 0.0]))
