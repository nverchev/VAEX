"""VAE / VAEX training entry point."""

import sys
from typing import TYPE_CHECKING, Any

from drytorch import Test, Trainer

from src.config import AllConfig, Experiment, get_trackers, hydra_main
from src.config.options import Models
from src.module.autoencoder import get_autoencoder
from src.train import get_learning_schema, get_loaders
from src.train.batch_norm import BatchNormUpdater
from src.train.hooks import register_checkpointing, register_early_stopping, register_pruning
from src.train.metrics_and_losses import get_vae_loss
from src.train.models import VAEEMAModel, VAEModel, clone_for_generation, load_extract_classifier_module
from src.utils.parallel import DistributedWorker

if TYPE_CHECKING:
    from optuna import Trial
else:
    Trial = Any


DEBUG_MODE = any(module in sys.modules for module in ('pydevd', 'debugpy', 'pdb')) or sys.gettrace() is not None


def train_vae(trial: Trial | None = None) -> None:
    cfg = Experiment.get_config()
    user_cfg = cfg.user
    vae_cfg = cfg.autoencoder
    architecture = get_autoencoder()
    if trial is None:
        model = VAEEMAModel(architecture, name=vae_cfg.name, device=user_cfg.device)
    else:
        model = VAEModel(architecture, name=vae_cfg.name, device=user_cfg.device)

    if vae_cfg.model.name == Models.VAEX:
        classifier = load_extract_classifier_module()
    else:
        classifier = None

    train_loader, eval_loader = get_loaders(
        batch_size=vae_cfg.train.batch_size_per_device,
        n_workers=user_cfg.n_workers,
        classifier=classifier
    )
    loss = get_vae_loss()
    schema = get_learning_schema(vae_cfg)
    trainer = Trainer(model, name=f'Trainer_{vae_cfg.name}', loader=train_loader, loss=loss, learning_schema=schema)
    test = Test(model, name=f'Test_{vae_cfg.name}', loader=eval_loader, metric=loss)
    if user_cfg.load:
        trainer.load_checkpoint(user_cfg.load)

    if not cfg.final:
        trainer.add_validation(eval_loader, f'Validation_{vae_cfg.name}')

    if not cfg.final and vae_cfg.train.early_stopping.active:
        cfg_early = vae_cfg.train.early_stopping
        register_early_stopping(trainer, window=cfg_early.window, patience=cfg_early.patience)

    if trial is not None:
        register_pruning(trainer, trial)
    else:
        register_checkpointing(trainer, user_cfg.checkpoint_every or 1)

    trainer.train_until(vae_cfg.train.n_epochs)
    test()
    if vae_cfg.objective.recalibrate_bn and isinstance(model, VAEEMAModel):
        generative = clone_for_generation(model)
        updater = BatchNormUpdater(
            generative,
            name=f'BatchNormUpdater_{vae_cfg.name}',
            loader=eval_loader,
        )
        updater()
        generative.save_state()

    return None


def setup_and_train(cfg: AllConfig) -> None:
    trackers = get_trackers(cfg, DEBUG_MODE)
    resume = cfg.user.load != 0 or cfg.autoencoder.model.name == Models.VAEX
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=resume, record=not DEBUG_MODE):
        train_vae()

    return None


@hydra_main
def main(cfg: AllConfig) -> None:
    n_processes = cfg.user.n_subprocesses
    if n_processes:
        DistributedWorker(setup_and_train, n_processes).spawn(cfg)
    else:
        setup_and_train(cfg)

    return None


if __name__ == '__main__':
    main()
