"""VAE / VAEX model profiling entry point."""

import pathlib
import torch

from src.config import AllConfig, Experiment, hydra_main
from src.data.dataset import get_datasets
from src.data.structures import Inputs, Outputs, Targets
from src.module.autoencoder import get_autoencoder
from src.train.metrics_and_losses import get_classifier_loss
from src.utils.profiling import profile_backward, profile_forward, save_parameters


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


def run_profiling(cfg: AllConfig) -> None:
    device = cfg.user.device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_split, _ = get_datasets()
    inputs, targets = train_split[0]
    batch_inputs = Inputs(
        image=inputs.image.unsqueeze(0).to(device),
        condition=inputs.condition.unsqueeze(0).to(device),
    )
    batch_targets = Targets(
        label=targets.label.unsqueeze(0).to(device),
        ref=inputs.image.unsqueeze(0).to(device),
    )

    model = get_autoencoder()
    model.to(device)
    profile_dir = _get_profiling_dir()
    save_parameters(model, profile_dir)

    loss_fn = get_classifier_loss()

    outputs = profile_forward(model, batch_inputs, device, profile_dir)
    if isinstance(outputs, Outputs) and getattr(outputs, 'model_epoch', None) is None:
        outputs.model_epoch = 0

    loss_dict = loss_fn.calculate(outputs, batch_targets)
    tensor_losses = [v for v in loss_dict.values() if isinstance(v, torch.Tensor) and v.requires_grad]
    if tensor_losses:
        loss = torch.stack(tensor_losses).sum()
        profile_backward(loss, device, profile_dir)

    return None


def setup_and_profile(cfg: AllConfig) -> None:
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True, record=False):
        run_profiling(cfg)

    return None


@hydra_main
def main(cfg: AllConfig) -> None:
    setup_and_profile(cfg)
    return None


if __name__ == '__main__':
    main()
