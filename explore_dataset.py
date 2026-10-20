"""Explore and inspect preprocessed dataset samples and attributes."""

import logging

import torch

from drytorch.lib import loading

from src.config import Experiment, ExperimentConfig, hydra_main
from src.data import Partitions, get_split
from src.trackers import get_trackers
from src.train.events import ImageEvent


@torch.inference_mode()
def explore_dataset() -> None:
    """Publish preprocessed dataset samples with their labels."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    if cfg_user.plot.use_train:
        partition = Partitions.train_val if cfg.final else Partitions.train
    else:
        partition = Partitions.test if cfg.final else Partitions.val

    split = get_split(partition)
    class_names = cfg.data.dataset.classes
    indices = cfg_user.plot.sample_indices
    if indices:
        sample_indices, (inputs, targets) = loading.take_from_dataset(split, indices=indices)
    else:
        sample_indices, (inputs, targets) = loading.take_from_dataset(
            split, n_samples=cfg_user.generate.batch_size, preserve_order=False
        )

    for i, image, label in zip(sample_indices, inputs.image.unbind(), targets.label.unbind(), strict=True):
        label_idx = int(label.item())
        label_name = class_names[label_idx] if label_idx < len(class_names) else f'class_{label_idx}'
        logging.info('Exploring Dataset for Sample %d (label: %s):', i, label_name)
        caption = label_name
        ImageEvent(image=image, caption=caption, title='original', folder=f'{partition.name}/sample_{i}')

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and launch dataset exploration."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        explore_dataset()

    return


if __name__ == '__main__':
    main()
