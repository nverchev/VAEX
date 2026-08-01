"""Generate random samples using the VAE and render them."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import AllConfig, Experiment, hydra_main
from src.train.models import load_extract_autoencoder_module


def to_numpy_img(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy image."""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img, 0.0, 1.0)


@torch.inference_mode()
def generate_samples() -> None:
    """Generate and visualize random samples from the latent space."""
    cfg = Experiment.get_config()
    batch_size = cfg.user.generate.batch_size
    save_dir = cfg.user.path.version_dir / "images" / cfg.name / "generated"
    save_dir.mkdir(parents=True, exist_ok=True)
    ema_module = load_extract_autoencoder_module()
    logging.info("Generating %d random samples using %s...", batch_size, type(ema_module).__name__)
    generated_images = ema_module.generate_sample(batch_size)
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    _, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))
    axes = np.atleast_2d(axes)
    for i in range(grid_size * grid_size):
        r, c = i // grid_size, i % grid_size
        ax = axes[r, c]
        if i < batch_size:
            ax.imshow(to_numpy_img(generated_images[i]))
            ax.set_title(f"Sample {i}")
        ax.axis('off')

    plt.tight_layout()
    save_path = save_dir / 'random_samples_grid.png'
    plt.savefig(save_path)
    plt.close()
    logging.info("Saved random samples to %s", save_path)
    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch the sample generation."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        generate_samples()

    return


if __name__ == "__main__":
    main()
