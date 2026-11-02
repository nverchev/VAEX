"""Tune the VAE with Optuna.

Each trial trains for ``n_epochs`` and scores the validation loss. The trials read the classifier
of the current experiment, so train it first with the same overrides. Trials must share ``sigma``
and ``c_kld``, which set the scale of the loss.

Usage::

    uv run tune_vae.py
    uv run tune_vae.py tune=architecture
"""

from collections.abc import Callable

import hydra
import optuna
import torch

from omegaconf import DictConfig

from drytorch.contrib.optuna import get_final_value, suggest_overrides
from drytorch.core.exceptions import ConvergenceError
from drytorch.core.registering import register_model, unregister_model

from src.config import ConfigPath, Experiment, ExperimentConfig, get_experiment_config
from src.lib.models import ClassifierModel, get_classifier_model, load_trained_state
from src.trackers import set_tuning_logging
from src.utils.tuning import create_study, impute_failed_trial, impute_pruned_trial
from train_vae import train_vae


def get_objective(tune_cfg: DictConfig, /) -> Callable[[optuna.Trial], float]:
    """Get the objective of the study: the validation loss of a short training.

    Args:
        tune_cfg: tuning configuration.

    Returns:
        A function training the VAE with the settings of a trial.
    """
    fixed_overrides = [*tune_cfg.overrides, f'autoencoder.train.n_epochs={tune_cfg.n_epochs}']
    classifier = _load_classifier(get_experiment_config(overrides=list(tune_cfg.overrides)))

    def objective(trial: optuna.Trial) -> float:
        overrides = [*fixed_overrides, *suggest_overrides(tune_cfg, trial)]
        cfg = get_experiment_config(overrides=overrides)
        exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
        with exp.create_run(record=False):
            register_model(classifier)
            try:
                train_vae(classifier, trial=trial)
            except optuna.TrialPruned:
                return impute_pruned_trial(trial)

            except ConvergenceError:
                return impute_failed_trial(trial)

            finally:
                unregister_model(classifier)
                if torch.accelerator.is_available():
                    torch.accelerator.empty_cache()

        return get_final_value(trial)

    return objective


@hydra.main(version_base=None, config_path=ConfigPath.TUNING.absolute(), config_name='defaults')
def main(tune_cfg: DictConfig) -> None:
    """Set up the study and launch the optimization."""
    set_tuning_logging()
    study = create_study(tune_cfg)

    # the trials of the study, from every worker and every run, stop at n_trials
    max_trials = optuna.study.MaxTrialsCallback(tune_cfg.tune.n_trials, states=None)
    study.optimize(get_objective(tune_cfg), n_trials=tune_cfg.tune.n_trials, callbacks=[max_trials])
    return


def _load_classifier(cfg: ExperimentConfig) -> ClassifierModel:
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=False):
        classifier = get_classifier_model()
        load_trained_state(classifier)
        unregister_model(classifier)

    return classifier


if __name__ == '__main__':
    main()
