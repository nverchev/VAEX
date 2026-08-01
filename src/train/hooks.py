"""Hooks to execute during training."""

from typing import Any

from drytorch import Trainer
from drytorch.lib.hooks import EarlyStoppingCallback, call_every, saving_hook
from drytorch.utils.averages import get_moving_average, get_trailing_mean


def register_checkpointing(trainer: Trainer, checkpoint_every: int | None) -> None:
    """Register the checkpointing hook."""
    if checkpoint_every:
        trainer.post_epoch_hooks.register(saving_hook.bind(call_every(checkpoint_every)))
    return


def register_early_stopping(trainer: Trainer, window: int, patience: int = 0) -> None:
    """Register the early stopping hook."""
    trainer.post_epoch_hooks.register(
        EarlyStoppingCallback(metric=trainer.objective, filter_fn=get_trailing_mean(window), patience=patience)
    )
    return


def register_pruning(trainer: Trainer, trial: Any) -> None:
    """Register the pruning hook."""
    from drytorch.contrib.optuna import TrialCallback

    prune_hook = TrialCallback(trial, metric=trainer.objective, filter_fn=get_moving_average())
    trainer.post_epoch_hooks.register(prune_hook)
    return
