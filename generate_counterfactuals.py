"""Generate counterfactuals of selected samples for each class and posterior relaxation, and render them.

Usage::

    uv run generate_counterfactuals.py
    uv run generate_counterfactuals.py "user.plot.sample_indices=[5]" user.trackers.wandb=true
"""

import logging

import torch

from drytorch.lib import loading

from src.config import Experiment, ExperimentConfig, hydra_main
from src.trackers import get_trackers
from src.data.dataset import get_split
from src.data.split import Partitions
from src.lib.models import get_autoencoder_model, get_classifier_model, load_trained_state
from src.utils.counterfactuals import get_counterfactual_sweep
from src.utils.visualization import publish_counterfactual_sweep


@torch.inference_mode()
def generate_counterfactuals() -> None:
    """Choose the samples, then generate and publish their counterfactuals towards every class."""
    cfg = Experiment.get_config()
    classifier = get_classifier_model()
    load_trained_state(classifier)
    vae = get_autoencoder_model()
    load_trained_state(vae, epoch=cfg.user.load or -1)
    partition = Partitions.test if cfg.final else Partitions.val
    split = get_split(partition)
    if cfg.user.plot.sample_indices:
        sample_indices, (samples, targets) = loading.take_from_dataset(
            split, indices=cfg.user.plot.sample_indices, device=classifier.device
        )
    else:
        sample_indices, (samples, targets) = loading.take_from_dataset(
            split, n_samples=cfg.user.generate.batch_size, preserve_order=False, device=classifier.device
        )

    relaxations = cfg.evaluation.relaxations
    logging.info('Generating counterfactuals for %d samples using %s...', len(sample_indices), vae.name)
    for index, image, label in zip(sample_indices, samples.image, targets.label.tolist(), strict=True):
        sweep = get_counterfactual_sweep(classifier, vae, image, label=label, relaxations=relaxations)
        publish_counterfactual_sweep(sweep, folder=f'{partition.name}/sample_{index}')

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and launch the counterfactual generation."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        generate_counterfactuals()

    return


if __name__ == '__main__':
    main()
