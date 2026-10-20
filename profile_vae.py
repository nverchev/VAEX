"""VAE / VAEX model profiling entry point."""

import math
import pathlib

import torch
from drytorch.lib.loading import take_from_dataset

from src.config import Experiment, ExperimentConfig, hydra_main
from src.data.dataset import get_splits
from src.data.structures import Outputs
from src.module.autoencoder import get_autoencoder
from src.train.objectives import get_vae_loss
from src.utils.profiling import log_parameter_counts, profile_backward, profile_forward


def run_profiling(cfg: ExperimentConfig) -> None:
    device = cfg.user.device
    if device == 'auto':
        device = torch.accelerator.current_accelerator(check_available=True) or torch.device('cpu')

    train_split, _ = get_splits()
    batch_size = cfg.autoencoder.train.batch_size
    _, (inputs, targets) = take_from_dataset(train_split, device=device, n_samples=batch_size)

    # uniform predictions stand in for those of the classifier
    n_classes = len(cfg.data.dataset.classes)
    inputs = inputs._replace(log_probs=torch.full((batch_size, n_classes), -math.log(n_classes), device=device))

    model = get_autoencoder()
    model.to(device)
    profile_dir = _get_profiling_dir()
    log_parameter_counts(model, profile_dir)

    loss_fn = get_vae_loss()
    outputs = profile_forward(model, inputs, device, profile_dir)
    if not isinstance(outputs, Outputs):
        raise TypeError('The VAE must return its outputs.')

    loss_dict = loss_fn.calculate(outputs, targets)
    tensor_losses = [v for v in loss_dict.values() if isinstance(v, torch.Tensor) and v.requires_grad]
    if tensor_losses:
        loss = torch.stack(tensor_losses).sum()
        profile_backward(loss, device, profile_dir)

    return


def setup_and_profile(cfg: ExperimentConfig) -> None:
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=False):
        run_profiling(cfg)

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    setup_and_profile(cfg)
    return


def _get_profiling_dir() -> pathlib.Path:
    exp = Experiment.get_current()
    if '@' in exp.run.id:
        day, time = exp.run.id.split('@')
        run_subdir = pathlib.Path(day) / time
    else:
        run_subdir = pathlib.Path(exp.run.id)

    profile_dir = exp.par_dir / 'profile' / exp.name / run_subdir
    profile_dir.mkdir(exist_ok=True, parents=True)
    return profile_dir


if __name__ == '__main__':
    main()
