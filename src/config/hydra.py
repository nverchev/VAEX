"""Hydra integration layer.

This module wires up Hydra's configuration composition with the project's
typed config hierarchy.  Two public helpers are exposed:

* :func:`hydra_main` — decorator that converts the Hydra ``DictConfig`` into a
  fully typed :class:`~src.config.specs.AllConfig` and passes it to the
  decorated entry-point function.

* :func:`get_config_all` — constructs an :class:`~src.config.specs.AllConfig`
  programmatically (without starting a Hydra run), used by the tuning script
  to instantiate fresh trial configs.
"""

import functools
import pathlib
from typing import cast
from collections.abc import Callable

import hydra
from hydra.conf import HydraConf
from hydra.core import config_store
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig

from src.config.environment import ConfigPath
from src.config.specs import AllConfig, HydraSettings
from src.config.experiment import update_variation


def hydra_main(func: Callable[[AllConfig], None]) -> Callable[[], None]:
    """Decorator that starts a Hydra run and passes a typed config to *func*.

    Usage::

        @hydra_main
        def main(cfg: AllConfig) -> None: ...

    The decorator:

    1. Reads ``configs/experiment/defaults.yaml`` and composes the full config.
    2. Casts the raw :class:`~omegaconf.DictConfig` to :class:`AllConfig`.
    3. Attaches Hydra runtime settings to ``cfg.user.hydra``.
    4. Calls :func:`~src.config.experiment.update_exp_name` to embed active
       overrides into the experiment name.

    Args:
        func: Entry-point function that accepts an :class:`AllConfig`.

    Returns:
        A no-argument callable suitable for use as ``__main__``.
    """

    @hydra.main(version_base=None, config_path=str(ConfigPath.EXPERIMENT.absolute()), config_name='defaults')
    @functools.wraps(func)
    def wrapper(dict_cfg: DictConfig) -> None:
        """Convert configuration to the stored object."""
        hydra_dict_cfg = HydraConfig.get()
        cfg = cast(AllConfig, OmegaConf.to_object(dict_cfg))
        cfg.user.hydra = get_hydra_settings(hydra_dict_cfg)
        overrides = hydra_dict_cfg.overrides.task
        update_variation(cfg, overrides)
        return func(cfg)

    return wrapper.__call__


def get_config_all(overrides: list[str] | None = None) -> AllConfig:
    """Compose and return an :class:`AllConfig` without starting a Hydra run.

    This is the recommended way to construct a config inside the tuning
    objective function, where each trial may use different overrides.

    Args:
        overrides: List of Hydra override strings, e.g.
            ``['experiment.train.learn.learning_rate=0.001']``.

    Returns:
        A fully typed and validated :class:`AllConfig`.
    """
    GlobalHydra.instance().clear()

    with hydra.initialize(version_base=None, config_path=ConfigPath.EXPERIMENT.relative()):
        dict_cfg = hydra.compose(config_name='defaults', overrides=overrides)
        try:
            hydra_dict_cfg = HydraConfig.get()
            hydra_settings = get_hydra_settings(hydra_dict_cfg)
        except ValueError:
            hydra_settings = HydraSettings()
            hydra_settings.output_dir = pathlib.Path('./outputs')
            hydra_settings.job_logging = OmegaConf.create()

        cfg = cast(AllConfig, OmegaConf.to_object(dict_cfg))
        cfg.user.hydra = hydra_settings

        if overrides is not None:
            update_variation(cfg, overrides)

        return cfg


def get_hydra_settings(dict_cfg: HydraConf) -> HydraSettings:
    """Get subset of hydra settings."""
    settings = HydraSettings()
    settings.output_dir = pathlib.Path(dict_cfg.runtime.output_dir)
    settings.job_logging = cast(DictConfig, dict_cfg.job_logging)
    return settings


cs = config_store.ConfigStore.instance()
cs.store(name='all_config', node=AllConfig)
