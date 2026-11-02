"""Utilities for hyperparameter tuning."""

import pathlib
import warnings

import numpy as np
import optuna

from omegaconf import DictConfig, OmegaConf
from optuna import visualization

from src.config import VERSION, get_experiment_config

__all__ = [
    'create_study',
    'get_past_final_values',
    'get_study_name',
    'impute_failed_trial',
    'impute_pruned_trial',
    'log_study_settings',
    'visualize_study',
]


def create_study(tune_cfg: DictConfig, /) -> optuna.Study:
    """Create the study a tuning configuration describes, or load it to continue.

    Args:
        tune_cfg: tuning configuration.

    Returns:
        The study, recording the tuning settings.

    Side Effects:
        Creates the folder of the study database.
    """
    pathlib.Path(tune_cfg.db_location).mkdir(exist_ok=True)
    pruner: optuna.pruners.BasePruner
    if tune_cfg.tune.use_pruner:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=tune_cfg.tune.n_startup_trials,
            n_warmup_steps=tune_cfg.tune.n_warmup_steps,
            interval_steps=tune_cfg.tune.interval_steps,
            n_min_trials=tune_cfg.tune.n_min_trials,
        )
    else:
        pruner = optuna.pruners.NopPruner()

    with warnings.catch_warnings():
        # the sampler is stable in practice; Optuna still labels it experimental
        warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.GPSampler(warn_independent_sampling=False)

    study_name = get_study_name(tune_cfg.tune.study_name, list(tune_cfg.overrides))
    study = optuna.create_study(
        study_name=study_name, storage=tune_cfg.storage, sampler=sampler, pruner=pruner, load_if_exists=True
    )
    log_study_settings(study, tune_cfg)
    return study


def get_past_final_values(trial: optuna.Trial, /) -> list[float]:
    """Get the final values of the completed trials that were not imputed.

    Args:
        trial: the current trial.

    Returns:
        The values of the past trials.

    Raises:
        TrialPruned: if fewer than ten past trials completed.
    """
    *past_trials, _current_trial = trial.study.get_trials(deepcopy=False)
    values = [
        past_trial.value
        for past_trial in past_trials
        if past_trial.state == optuna.trial.TrialState.COMPLETE
        and past_trial.value is not None
        and not past_trial.user_attrs.get('imputed', False)
    ]
    if len(values) < 10:
        raise optuna.TrialPruned()

    return values


def get_study_name(tuning_scheme: str, overrides: list[str]) -> str:
    """Get the name of a study from the configuration its trials start from.

    Args:
        tuning_scheme: name of the group of tuned settings.
        overrides: fixed overrides of every trial.

    Returns:
        The version, the variation, the overrides and the scheme, joined.
    """
    variation = get_experiment_config().resolved_variation
    short_overrides = [override.rsplit('.', maxsplit=1)[-1] for override in sorted(overrides)]
    return '_'.join([f'v{VERSION}', variation, *short_overrides, tuning_scheme])


def impute_failed_trial(trial: optuna.Trial, /) -> float:
    """Get the value of a failed trial: the worst of the completed trials.

    Args:
        trial: the failed trial.

    Returns:
        The imputed value.
    """
    past_final_values = get_past_final_values(trial)
    worst_fn = min if trial.study.direction == optuna.study.StudyDirection.MAXIMIZE else max
    trial.set_user_attr('imputed', True)
    return worst_fn(past_final_values)


def impute_pruned_trial(trial: optuna.Trial, /) -> float:
    """Get the value of a pruned trial: a poor quartile of the completed trials.

    Args:
        trial: the pruned trial.

    Returns:
        The imputed value.
    """
    past_final_values = get_past_final_values(trial)
    percentile = 75 if trial.study.direction == optuna.study.StudyDirection.MINIMIZE else 25
    trial.set_user_attr('imputed', True)
    return float(np.percentile(past_final_values, percentile))


def log_study_settings(study: optuna.Study, tune_cfg: DictConfig) -> None:
    """Record the tuning settings in the study.

    Args:
        study: the study.
        tune_cfg: tuning configuration.

    Raises:
        ValueError: if the configuration is not a mapping.
    """
    cfg_dict = OmegaConf.to_container(tune_cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError('The tuning configuration must be a dictionary.')

    for key, value in cfg_dict.items():
        study.set_user_attr(str(key), value)

    return


def visualize_study(study: optuna.Study, /, *, renderer: str = '') -> None:
    """Show the plots of a study.

    Args:
        study: the study.
        renderer: the Plotly renderer.
    """
    visualization.plot_optimization_history(study).show(renderer=renderer)
    visualization.plot_slice(study).show(renderer=renderer)
    visualization.plot_parallel_coordinate(study).show(renderer=renderer)
    visualization.plot_param_importances(study).show(renderer=renderer)
    return
