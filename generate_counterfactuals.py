"""Generate counterfactuals of selected samples for each class and posterior relaxation, and render them.

It also renders the counterfactuals of every image in ``<data_dir>/custom_images``, faces outside
the dataset aligned as the CelebA images.

Usage::

    uv run generate_counterfactuals.py
    uv run generate_counterfactuals.py "user.plot.sample_indices=[5]" user.trackers.wandb=true
"""

import logging
from typing import Final

import torch

from drytorch.lib import loading

from src.config import Experiment, ExperimentConfig, hydra_main
from src.trackers import get_trackers
from src.data.dataset import get_split
from src.data.preprocess import get_preprocessor
from src.data.split import Partitions
from src.data.structures import Inputs
from src.lib.models import ClassifierModel, VAEModel, load_autoencoder_model, get_classifier_model, load_trained_state
from src.utils.counterfactuals import get_counterfactual_sweep
from src.utils.visualization import publish_counterfactual_sweep

IMAGE_SUFFIXES: Final = ('.jpg', '.jpeg', '.png')


@torch.inference_mode()
def generate_counterfactuals() -> None:
    """Choose the samples, then generate and publish their counterfactuals towards every class."""
    cfg = Experiment.get_config()
    classifier = get_classifier_model()
    load_trained_state(classifier)
    vae = load_autoencoder_model()
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

    generate_custom_counterfactuals(classifier, vae)
    return


def generate_custom_counterfactuals(classifier: ClassifierModel, vae: VAEModel) -> None:
    """Generate and publish the counterfactuals of every image in the folder of custom images.

    The class of each image is the one the classifier predicts.

    Args:
        classifier: the trained classifier.
        vae: the trained VAE.
    """
    cfg = Experiment.get_config()
    folder = cfg.user.path.custom_images_dir
    paths = sorted(path for path in folder.glob('*') if path.suffix.lower() in IMAGE_SUFFIXES)
    logging.info('Generating counterfactuals for %d custom images in %s...', len(paths), folder)
    preprocessor = get_preprocessor()
    for path in paths:
        image = preprocessor.load_image(path).to(classifier.device)
        label = int(classifier.log_probabilities(Inputs(image=image.unsqueeze(0))).argmax(dim=1).item())
        sweep = get_counterfactual_sweep(classifier, vae, image, label=label, relaxations=cfg.evaluation.relaxations)
        publish_counterfactual_sweep(sweep, folder=f'custom/{path.stem}')

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
