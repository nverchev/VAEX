"""Recalibrate BatchNorm statistics for an already trained checkpoint."""

from drytorch import Experiment as _Experiment

from src.config import AllConfig, Experiment, get_trackers, hydra_main
from src.config.options import Models
from src.module.autoencoder import get_autoencoder
from src.train import get_loaders
from src.train.batch_norm import BatchNormUpdater
from src.train.models import VAEEMAModel, clone_for_generation, load_extract_classifier_module


def recalibrate() -> None:
    cfg = Experiment.get_config()
    vae_cfg = cfg.autoencoder
    model = VAEEMAModel(get_autoencoder(), name=vae_cfg.name, device=cfg.user.device)
    model.load_state(cfg.user.load if cfg.user.load != 0 else -1)

    classifier = load_extract_classifier_module() if vae_cfg.model.name == Models.VAEX else None
    _, eval_loader = get_loaders(
        batch_size=vae_cfg.train.batch_size_per_device,
        n_workers=cfg.user.n_workers,
        classifier=classifier,
    )

    generative = clone_for_generation(model)
    updater = BatchNormUpdater(generative, name=f'BatchNormUpdater_{vae_cfg.name}', loader=eval_loader)
    updater()
    generative.save_state()
    return None


@hydra_main
def main(cfg: AllConfig) -> None:
    trackers = get_trackers(cfg, False)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, record=True):
        recalibrate()

    return None


if __name__ == '__main__':
    main()
