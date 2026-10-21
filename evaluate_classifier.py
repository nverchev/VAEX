"""Evaluate the trained classifier on the evaluation partition.

The evaluation partition is the test set only with ``final=true``, the validation set otherwise.

Usage::

    uv run evaluate_classifier.py
    uv run evaluate_classifier.py final=true
"""

from drytorch import Test

from src.config import Experiment, ExperimentConfig, hydra_main
from src.config.environment import DEBUG_MODE
from src.lib import get_loaders
from src.lib.models import get_classifier_model, load_trained_state
from src.lib.objectives import get_accuracy
from src.trackers import get_trackers


def evaluate_classifier() -> None:
    """Test the accuracy of the trained classifier."""
    cfg = Experiment.get_config()
    model = get_classifier_model()
    load_trained_state(model, epoch=cfg.user.load or -1)
    _, eval_loader = get_loaders(
        batch_size=cfg.classification.train.batch_size_per_device, n_workers=cfg.user.n_workers
    )
    Test(model, name='Accuracy', loader=eval_loader, metric=get_accuracy())()
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and evaluate the classifier."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        evaluate_classifier()

    return


if __name__ == '__main__':
    main()
