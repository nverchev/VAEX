"""Hooks to execute during training."""

from typing import Any

from drytorch.lib.training import TrainerBuilder
from drytorch.utils.averages import get_trailing_mean

__all__ = [
    'register_checkpointing',
    'register_early_stopping',
    'register_pruning',
]


def register_checkpointing(builder: TrainerBuilder[Any, Any, Any], /, *, checkpoint_every: int) -> None:
    """Register the checkpointing hook.

    Args:
        builder: trainer builder to register the saving hook to.
        checkpoint_every: save frequency in epochs; ``0`` adds no saving.
    """
    if checkpoint_every:
        builder.add_saving(checkpoint_every)

    return


def register_early_stopping(builder: TrainerBuilder[Any, Any, Any], /, *, patience: int = 0, window: int) -> None:
    """Register the early stopping hook.

    Args:
        builder: trainer builder to register early stopping to.
        patience: number of evaluations without improvement before stopping.
        window: trailing window size for the smoothed metric.
    """
    builder.add_early_stopping(
        patience=patience,
        watch=builder.watch(filter_fn=get_trailing_mean(window)),
    )
    return


def register_pruning(builder: TrainerBuilder[Any, Any, Any], /, trial: Any = None, *, window: int) -> None:
    """Register the pruning hook.

    Args:
        builder: trainer builder to register pruning callback to.
        trial: Optuna trial object.
        window: number of last validations the pruner averages; ``1`` reads the last one.
    """
    from drytorch.contrib.optuna import TrialCallback

    builder.add_post_epoch_hook(TrialCallback(trial, watch=builder.watch(filter_fn=get_trailing_mean(window))))
    return
