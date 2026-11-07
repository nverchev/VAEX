"""Generate counterfactuals of samples the classifier gets wrong, towards every class, and render them.

Usage::

    uv run generate_misclassified_counterfactuals.py final=true
"""

import logging

import torch

from drytorch import DataLoader
from drytorch.lib import loading
from drytorch.utils.apply_ops import apply_to

from src.config import Experiment, ExperimentConfig, hydra_main
from src.data.dataset import get_split
from src.data.split import DataSplit, Partitions, Subset
from src.trackers import get_trackers
from src.lib.loaders import get_calibration_loader
from src.lib.models import ClassifierModel, load_autoencoder_model, get_classifier_model, load_trained_state
from src.utils.counterfactuals import get_counterfactual_sweep
from src.utils.visualization import publish_counterfactual_sweep


@torch.inference_mode()
def generate_misclassified_counterfactuals() -> None:
    """Choose samples the classifier gets wrong, then generate and publish their counterfactuals."""
    cfg = Experiment.get_config()
    classifier = get_classifier_model()
    load_trained_state(classifier)
    vae = load_autoencoder_model()
    vae.set_calibration(get_calibration_loader(classifier))
    partition = Partitions.test if cfg.final else Partitions.val
    split = get_split(partition)
    misclassified = Subset(split, _get_misclassified(classifier, split))
    subset_indices, (samples, targets) = loading.take_from_dataset(
        misclassified, n_samples=cfg.user.generate.batch_size, preserve_order=False, device=classifier.device
    )
    sample_indices = [misclassified.indices[index] for index in subset_indices]
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
        generate_misclassified_counterfactuals()

    return


def _get_misclassified(classifier: ClassifierModel, split: DataSplit) -> list[int]:
    batch_size = Experiment.get_config().classification.train.batch_size
    misclassified: list[torch.Tensor] = []
    for inputs, targets in DataLoader(dataset=split, batch_size=batch_size, pin_memory=False):
        inputs, targets = apply_to((inputs, targets), classifier.device)
        misclassified.append(classifier.log_probabilities(inputs).argmax(dim=1) != targets.label)

    return torch.cat(misclassified).nonzero().squeeze(1).tolist()


if __name__ == '__main__':
    main()
