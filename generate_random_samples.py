"""Generate random samples using the VAE and render them."""

import logging

import torch

from src.config import Experiment, ExperimentConfig, hydra_main
from src.trackers import get_trackers
from src.train.events import ImageEvent
from src.train.models import get_autoencoder_model, load_trained_state


@torch.inference_mode()
def generate_samples() -> None:
    """Generate random samples from the priors and publish each.

    Each sample has random probabilities of the classes as its condition.
    """
    cfg = Experiment.get_config()
    batch_size = cfg.user.generate.batch_size
    vae = get_autoencoder_model()
    load_trained_state(vae, epoch=cfg.user.load or -1)
    vae_module = vae.module.eval()
    logging.info('Generating %d random samples using %s...', batch_size, vae.name)
    condition = vae_module.get_random_condition(batch_size)
    generated_images = vae_module.generate(condition)
    for i, image in enumerate(generated_images):
        ImageEvent(image=image, title=f'sample_{i}', folder='generated')

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and launch the sample generation."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        generate_samples()

    return


if __name__ == '__main__':
    main()
