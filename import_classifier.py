"""Copy a trained classifier into the current experiment instead of training it again."""

import logging
import pathlib
import shutil

from src.config import AllConfig, Experiment, hydra_main


def latest_checkpoint(experiment_dir: pathlib.Path, model_name: str) -> pathlib.Path:
    """Return the highest-numbered epoch saved for a model across the experiment's runs.

    Args:
        experiment_dir: checkpoint folder of the experiment holding the trained model.
        model_name: name the model was registered under.

    Raises:
        FileNotFoundError: if the experiment holds no checkpoint for that model.
    """
    saved = list(experiment_dir.glob(f'*/*/{model_name}/epoch_*'))
    if not saved:
        raise FileNotFoundError(f'no {model_name} checkpoint under {experiment_dir}')

    return max(saved, key=lambda path: int(path.name.removeprefix('epoch_')))


@hydra_main
def main(cfg: AllConfig) -> None:
    source_name = cfg.user.path.source_experiment
    if not source_name:
        raise ValueError('set user.path.source_experiment to the experiment holding the trained classifier')

    version_dir = cfg.user.path.version_dir
    model_name = cfg.classification.name
    source = latest_checkpoint(pathlib.Path(version_dir) / 'checkpoints' / source_name, model_name)

    exp = Experiment(cfg, name=cfg.name, par_dir=version_dir, tags=cfg.tags)
    run = exp.create_run(resume=False, record=True)
    with run:
        day, time = run.id.split('@')
        target = pathlib.Path(version_dir) / 'checkpoints' / cfg.name / day / time / model_name / source.name
        shutil.copytree(source, target, dirs_exist_ok=True)
        logging.info('copied %s -> %s', source, target)

    return None


if __name__ == '__main__':
    main()
