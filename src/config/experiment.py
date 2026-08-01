"""Experiment management and tracker initialisation."""

import os
import warnings

from hydra.core import utils as hydra_utils

import drytorch

from drytorch.core import tracking
from drytorch.trackers import logging

from src.config.specs import AllConfig


class Experiment(drytorch.Experiment[AllConfig]):
    """Specify specification class for configurations."""


def get_trackers(cfg: AllConfig, debug_mode: bool) -> list[tracking.Tracker]:
    """Instantiate the experiment trackers requested in *cfg*.

    Trackers are only activated outside debug mode.  In debug mode an empty
    list is returned so that no files or remote metrics are written.

    Args:
        cfg: Root configuration object.
        debug_mode: When ``True``, skip all tracker initialisation.

    Returns:
        A list of initialised :class:`drytorch.core.tracking.Tracker` objects
        ready to be passed to :meth:`drytorch.Experiment.trackers.subscribe`.
    """
    cfg_trackers = cfg.user.trackers
    cfg_hydra = cfg.user.hydra
    hydra_utils.configure_log(cfg_hydra.job_logging)
    drytorch.init_trackers(mode='hydra')
    optional_tracker_list: list[tracking.Tracker] = []
    if debug_mode:
        return optional_tracker_list

    if cfg_trackers.wandb:
        try:
            import wandb  # ty:ignore[unresolved-import]

            from drytorch.trackers.wandb import Wandb

            optional_tracker_list.append(Wandb(settings=wandb.Settings(project=cfg.project)))
        except ImportError as e:
            warnings.warn(f'Wandb not installed, skipping. {e}', stacklevel=2)

    if cfg_trackers.hydra:
        from drytorch.trackers.hydra import HydraLink

        optional_tracker_list.append(HydraLink(hydra_dir=cfg.user.hydra.output_dir))

    if cfg_trackers.csv:
        from drytorch.trackers.csv import CSVDumper

        optional_tracker_list.append(CSVDumper())

    if cfg_trackers.tensorboard:
        from drytorch.trackers.tensorboard import TensorBoard

        optional_tracker_list.append(TensorBoard())

    if cfg_trackers.sqlalchemy:
        try:
            import sqlalchemy  # ty:ignore[unresolved-import]

            from drytorch.trackers.sqlalchemy import SQLConnection

        except ImportError as e:
            warnings.warn(f'SQLAlchemy not installed, skipping. {e}', stacklevel=2)

        else:
            engine_path = cfg.user.path.version_dir / 'metrics.db'
            cfg.user.path.version_dir.mkdir(exist_ok=True)
            engine = sqlalchemy.create_engine(f'sqlite:///{engine_path}')
            optional_tracker_list.append(SQLConnection(engine=engine))

    return optional_tracker_list


def update_variation(cfg: AllConfig, overrides: list[str]) -> None:
    """Embed active Hydra overrides into the experiment variation name.

    User-specific overrides do not affect the experiment and are filtered out.
    The logic for the final flag is handled by the Experiment class.

    Args:
        cfg: Root configuration object (mutated in place).
        overrides: Raw Hydra task override strings.
    """
    filtered_overrides = [
        override
        for override in overrides
        if override.split('.')[0] != 'user' and override.split('=')[0] not in ('final', 'variation')
    ]
    sorted_overrides = sorted(filtered_overrides)
    cfg.variation = '_'.join([cfg.resolved_variation, *sorted_overrides]).replace('/', '_')
    cfg.tags = sorted_overrides
    return


def set_tuning_logging() -> None:
    """Configure minimal console logging for parallel Optuna workers.

    When ``PARALLEL_SEQ`` is set in the environment (indicating a spawned
    worker), all default trackers are replaced by a plain built-in logger
    so that only test-level messages are printed.  Otherwise, Hydra's
    minimal mode is used.
    """
    if 'PARALLEL_SEQ' in os.environ:
        drytorch.remove_all_default_trackers()
        drytorch.extend_default_trackers([logging.BuiltinLogger()])
        logging.set_verbosity(logging.INFO_LEVELS.test)
    else:
        drytorch.init_trackers(mode='minimal')

    return
