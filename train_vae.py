"""Train the VAEX autoencoder conditioned on the trained classifier.

Usage::

    uv run train_vae.py
    uv run train_vae.py autoencoder.model.architecture.inference_block=mobile   # VAEXcat
"""

from drytorch import Trainer

from src.config import Experiment, ExperimentConfig, hydra_main
from src.trackers import get_trackers, save_learning_curves
from src.config.environment import DEBUG_MODE
from src.lib import get_learning_schema, get_loaders
from src.lib.hooks import register_checkpointing, register_early_stopping
from src.lib.objectives import get_vae_loss
from src.lib.models import get_autoencoder_model, get_classifier_model, load_trained_state
from src.utils.parallel import DistributedWorker


def train_vae() -> None:
    """Train the autoencoder and save it."""
    cfg = Experiment.get_config()
    user_cfg = cfg.user
    vae_cfg = cfg.autoencoder
    model = get_autoencoder_model(mixed_precision=vae_cfg.train.mixed_precision)
    classifier = get_classifier_model()
    load_trained_state(classifier)
    train_loader, eval_loader = get_loaders(
        batch_size=vae_cfg.train.batch_size_per_device,
        n_workers=user_cfg.n_workers,
        classifier=classifier,
    )
    loss = get_vae_loss()
    builder = Trainer.builder(
        model,
        learning_schema=get_learning_schema(vae_cfg),
        loader=train_loader,
        loss=loss,
    )
    if not cfg.final:
        builder.add_validation(eval_loader)

    if not cfg.final and vae_cfg.train.early_stopping.active:
        cfg_early = vae_cfg.train.early_stopping
        register_early_stopping(builder, window=cfg_early.window, patience=cfg_early.patience)

    register_checkpointing(builder, checkpoint_every=user_cfg.checkpoint_every)
    trainer = builder.build()
    if user_cfg.load:
        trainer.load_checkpoint(user_cfg.load)

    trainer.train_until(vae_cfg.train.n_epochs)
    trainer.save_checkpoint()
    return


def setup_and_train(cfg: ExperimentConfig) -> None:
    """Set up the experiment and train the autoencoder.

    Args:
        cfg: root configuration object.
    """
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    # resume the run holding the classifier: the last run of the experiment, or the one user.run_id names
    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        train_vae()
        save_learning_curves(trackers, cfg.autoencoder.model.name, show=cfg.user.plot.show)

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Entry point, spawning the distributed workers when ``n_subprocesses`` is set."""
    n_processes = cfg.user.n_subprocesses
    if n_processes:
        DistributedWorker(setup_and_train, n_processes).spawn(cfg)
    else:
        setup_and_train(cfg)

    return


if __name__ == '__main__':
    main()
