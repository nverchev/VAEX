"""Show the results of a study.

Usage::

    uv run plot_tuning_results.py
    uv run plot_tuning_results.py tune=architecture
"""

import hydra
import optuna

from omegaconf import DictConfig

from src.config import ConfigPath
from src.utils.tuning import get_study_name, visualize_study


@hydra.main(version_base=None, config_path=ConfigPath.TUNING.absolute(), config_name='defaults')
def main(tune_cfg: DictConfig) -> None:
    """Load the study and show its plots."""
    study_name = get_study_name(tune_cfg.tune.study_name, list(tune_cfg.overrides))
    study = optuna.load_study(study_name=study_name, storage=tune_cfg.storage)
    visualize_study(study, renderer=tune_cfg.renderer)
    return


if __name__ == '__main__':
    main()
