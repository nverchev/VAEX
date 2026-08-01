"""Explore and inspect preprocessed dataset samples and attributes."""

import logging
import random
from collections.abc import Sized

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.config import AllConfig, Experiment, hydra_main
from src.data import Partitions, get_dataset


@torch.inference_mode()
def explore_dataset() -> None:
    """Visualize preprocessed dataset samples and save image artefacts."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    save_dir_base = cfg_user.path.version_dir / 'images' / 'dataset'

    if cfg_user.plot.use_train:
        partition = Partitions.train_val if cfg.final else Partitions.train
    else:
        partition = Partitions.test if cfg.final else Partitions.val

    dataset = get_dataset(partition)
    class_names = cfg.data.dataset.classes
    sample_indices = cfg_user.plot.sample_indices
    if not sample_indices:
        assert isinstance(dataset, Sized)
        sample_indices = random.sample(range(len(dataset)), min(8, len(dataset)))

    for i in sample_indices:
        assert isinstance(dataset, Sized)
        if i >= len(dataset):
            raise ValueError(f'Index {i} is too large for the selected dataset of length {len(dataset)}')

        inputs, targets = dataset[i]
        img_tensor = inputs.image
        img_np = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0.0, 1.0)
        label_idx = int(targets.label.item()) if targets.label is not None else 0
        label_name = class_names[label_idx] if label_idx < len(class_names) else f'class_{label_idx}'
        logging.info('Exploring Dataset for Sample %d (label: %s):', i, label_name)

        save_dir = save_dir_base / f'sample_{i}'
        save_dir.mkdir(parents=True, exist_ok=True)

        img_pil = Image.fromarray((img_np * 255.0).astype(np.uint8))
        img_pil.save(save_dir / 'sample.png')

        plt.figure(figsize=(4, 4))
        plt.imshow(img_np)
        plt.title(f'Sample {i} ({label_name})')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_dir / 'sample_plot.png', bbox_inches='tight')
        plt.close()

    return None


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch dataset exploration."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(record=False):
        explore_dataset()

    return None


if __name__ == '__main__':
    main()
