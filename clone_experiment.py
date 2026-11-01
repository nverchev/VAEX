"""Clone an experiment under the name that overrides form, to reuse its earlier phases.

The run name includes every override, so a later phase run with new overrides
looks for the earlier phases under a new name. Cloning the experiment that holds
them under that name lets the later phase resume it without training them again.
The metric database follows the folders, so that the plots keep the earlier results.

Usage::

    uv run clone_experiment.py autoencoder.train.n_epochs=100
    uv run clone_experiment.py --from published autoencoder.train.n_epochs=100
    uv run clone_experiment.py --from other_branch
"""

import argparse

from drytorch.core import exceptions
from drytorch.utils import local_ops

from src.config.hydra import get_experiment_config
from src.trackers import get_metric_database


def main() -> None:
    """Clone an experiment under the name of the configuration with the overrides."""
    parser = argparse.ArgumentParser(description='Clone an experiment under the name that overrides form.')
    parser.add_argument(
        '--from', dest='from_', metavar='NAME', help='experiment to clone; by default, the current variation'
    )
    parser.add_argument('overrides', nargs='*', help='overrides of the new experiment, as for the training scripts')
    args = parser.parse_args()
    cfg = get_experiment_config(overrides=args.overrides)
    from_name = args.from_
    if from_name is None:
        if cfg.final:
            parser.error('a final experiment needs --from: name the final experiment to clone')

        from_name = get_experiment_config().name

    if from_name.endswith('_final') != cfg.final:
        parser.error(f'{from_name} and {cfg.name} mix a final and a development experiment')

    if from_name == cfg.name:
        parser.error(f'nothing to clone: {from_name} is already the experiment of these settings')

    par_dir = cfg.user.path.version_dir
    print(f'Cloning {from_name} as {cfg.name} in {par_dir}...')
    local_ops.clone_experiment_data(par_dir, from_name, cfg.name)
    database = get_metric_database(cfg)
    if database is not None:
        try:
            database.clone_experiment(from_name, cfg.name)
        except exceptions.TrackerError as err:
            print(f'Metric database not updated: {err}')

        database.close()

    print('Done.')
    return


if __name__ == '__main__':
    main()
