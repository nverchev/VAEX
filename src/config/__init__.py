"""Package for configuring the experiment."""

from src.config.specs import AllConfig
from src.config.hydra import hydra_main, get_config_all
from src.config.experiment import Experiment, get_trackers, set_tuning_logging
from src.config.environment import VERSION, ConfigPath
from src.config.torch import ActClass, NormClass

__all__ = [
    'VERSION',
    'ActClass',
    'AllConfig',
    'ConfigPath',
    'Experiment',
    'NormClass',
    'get_config_all',
    'get_trackers',
    'hydra_main',
    'set_tuning_logging',
]
