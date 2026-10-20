"""Learning curves that the plotting trackers recorded during training."""

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from torch import distributed as dist

from drytorch.core import tracking
from drytorch.trackers import base_classes


__all__ = [
    'save_learning_curves',
]


def save_learning_curves(
    trackers: Iterable[tracking.Tracker],
    model_name: str,
    *,
    show: bool = False,
    source_names: Iterable[str] = ('Trainer', 'Validation'),
) -> None:
    """Save the learning curves of a model that the plotting trackers recorded.

    Each plotting tracker (matplotlib, plotly) writes one figure per metric in the
    folder of the run; with ``show``, matplotlib also opens all the metrics in one
    window and plotly opens each figure in the browser. Other trackers are skipped.

    Args:
        trackers: the trackers of the run.
        model_name: the name of the model.
        show: whether to also display the figures, which needs a display.
        source_names: the sources whose curves to plot.

    Side Effects:
        Writes the figures in the folder of the run and, with ``show``, displays them.
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return  # trackers record only in the main process

    for tracker in trackers:
        if not isinstance(tracker, base_classes.BasePlotter):
            continue

        tracker.save_figures(model_name, source_names=source_names)
        if not show:
            continue

        # matplotlib shows all the metrics in one window
        if isinstance(tracker, _GridPlotterProtocol):
            tracker.plot_grid(model_name, source_names=source_names)
        else:
            tracker.plot(model_name, source_names=source_names)

    return


@runtime_checkable
class _GridPlotterProtocol(Protocol):
    def plot_grid(
        self,
        model_name: str,
        *,
        metric_names: Iterable[str] = ...,
        source_names: Iterable[str] = ...,
        start_epoch: int = ...,
    ) -> None:
        """Render the metrics of the model as subplots of one figure."""
