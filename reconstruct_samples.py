"""Reconstruct samples from the dataset."""

import logging

import torch

from drytorch.lib import loading

from src.config import Experiment, ExperimentConfig, hydra_main
from src.trackers import get_trackers
from src.lib.events import ImageEvent
from src.data import Inputs, Partitions, get_split
from src.lib.models import load_autoencoder_model, get_classifier_model, load_trained_state


@torch.inference_mode()
def reconstruct_samples() -> None:
    """Reconstruct selected dataset samples and publish each next to its original."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    if cfg_user.plot.use_train:
        partition = Partitions.train_val if cfg.final else Partitions.train
    else:
        partition = Partitions.test if cfg.final else Partitions.val

    split = get_split(partition)
    vae_model = load_autoencoder_model()
    classifier = get_classifier_model()
    load_trained_state(classifier)
    device = vae_model.device
    if cfg_user.plot.sample_indices:
        sample_indices, (samples, targets) = loading.take_from_dataset(
            split, indices=cfg_user.plot.sample_indices, device=device
        )
    else:
        sample_indices, (samples, targets) = loading.take_from_dataset(
            split, n_samples=cfg_user.generate.batch_size, preserve_order=False, device=device
        )

    logging.info('Reconstructing %d samples using %s...', len(sample_indices), vae_model.name)
    inputs = Inputs(image=samples.image, condition=targets.label.float())
    inputs = inputs._replace(log_probs=classifier.log_probabilities(inputs))
    outputs = vae_model.reconstruct(inputs)
    original_captions = classifier.describe_predictions(inputs.image)
    recon_captions = classifier.describe_predictions(outputs.recon)
    for position, index in enumerate(sample_indices):
        original = inputs.image[position].clamp(0, 1)
        recon = outputs.recon[position].clamp(0, 1)
        label = cfg.data.dataset.classes[int(targets.label[position])]
        logging.info('Reconstructing Sample %d (label: %s):', index, label)
        folder = f'{partition.name}/sample_{index}'
        ImageEvent(image=original, title='original', caption=f'{label}, {original_captions[position]}', folder=folder)
        ImageEvent(image=recon, title='reconstruction', caption=recon_captions[position], folder=folder)
        ImageEvent(image=(original - recon).abs(), title='difference', caption='Absolute difference', folder=folder)

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and launch the sample reconstruction."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        reconstruct_samples()

    return


if __name__ == '__main__':
    main()
