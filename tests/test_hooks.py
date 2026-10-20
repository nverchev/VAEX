"""Tests for the hook helpers."""

from unittest.mock import MagicMock

from src.train.hooks import register_checkpointing, register_early_stopping


def test_checkpointing_saves_at_the_given_frequency() -> None:
    """The builder saves every given number of epochs."""
    builder = MagicMock()

    register_checkpointing(builder, checkpoint_every=5)

    builder.add_saving.assert_called_once_with(5)


def test_zero_frequency_saves_nothing() -> None:
    """A frequency of zero registers no saving."""
    builder = MagicMock()

    register_checkpointing(builder, checkpoint_every=0)

    builder.add_saving.assert_not_called()


def test_early_stopping_watches_the_smoothed_loss() -> None:
    """Early stopping is added with its patience and a watch over a trailing mean."""
    builder = MagicMock()

    register_early_stopping(builder, patience=3, window=4)

    builder.add_early_stopping.assert_called_once_with(patience=3, watch=builder.watch.return_value)
    builder.watch.assert_called_once()
