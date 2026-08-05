"""Reconstruct samples from the dataset."""

import logging
import random

from collections.abc import Sized

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.config import AllConfig, Experiment, hydra_main
from src.config.options import Models
from src.data import Inputs, Partitions, get_dataset
from src.train.models import load_extract_autoencoder_module, load_extract_classifier_module


@torch.inference_mode()
def reconstruct_samples() -> None:
    """Reconstruct and visualize selected dataset samples."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    save_dir_base = cfg_user.path.version_dir / 'images' / cfg.name / 'reconstructed'
    if cfg_user.plot.use_train:
        partition = Partitions.train_val if cfg.final else Partitions.train
    else:
        partition = Partitions.test if cfg.final else Partitions.val

    dataset = get_dataset(partition)
    assert isinstance(dataset, Sized)
    vae_module = load_extract_autoencoder_module()
    if cfg.autoencoder.model.name == Models.VAEX:
        classifier = load_extract_classifier_module()
    else:
        classifier = None

    device = next(vae_module.parameters()).device
    sample_indices = cfg_user.plot.sample_indices

    if not sample_indices:
        sample_indices = random.sample(range(len(dataset)), min(8, len(dataset)))

    for i in sample_indices:
        if i >= len(dataset):
            raise ValueError(f'Index {i} is too large for the selected dataset of length {len(dataset)}')

        inputs_item, target_item = dataset[i]
        img_tensor = inputs_item.image.unsqueeze(0).to(device)
        cond_tensor = (
            target_item.label.unsqueeze(0).to(device).float()
            if target_item.label is not None
            else torch.empty(0, device=device)
        )
        inputs = Inputs(image=img_tensor, condition=cond_tensor)
        if classifier is not None:
            logits = classifier(inputs)
            inputs = inputs._replace(logits=logits)

        outputs = vae_module(inputs)
        recon_img = outputs.recon
        original_np = img_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        recon_np = recon_img[0].detach().cpu().numpy().transpose(1, 2, 0)
        original_np = np.clip(original_np, 0.0, 1.0)
        recon_np = np.clip(recon_np, 0.0, 1.0)
        label_idx = int(target_item.label.item()) if target_item.label is not None else 0
        logging.info('Reconstructing Sample %d (label index: %d):', i, label_idx)
        save_dir = save_dir_base / f'sample_{i}'
        save_dir.mkdir(parents=True, exist_ok=True)
        _, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(original_np)
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(recon_np)
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        diff_np = np.abs(original_np - recon_np)
        axes[2].imshow(diff_np)
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')
        plt.tight_layout()
        comparison_path = save_dir / 'comparison.png'
        plt.savefig(comparison_path, bbox_inches='tight')
        plt.close()
        orig_img_pil = Image.fromarray((original_np * 255.0).astype(np.uint8))
        recon_img_pil = Image.fromarray((recon_np * 255.0).astype(np.uint8))
        orig_img_pil.save(save_dir / 'original.png')
        recon_img_pil.save(save_dir / 'reconstructed.png')

    return None


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch the sample reconstruction."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        reconstruct_samples()

    return None


if __name__ == '__main__':
    main()
