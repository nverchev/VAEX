"""Train the classifier whose probabilities condition the VAE."""

from drytorch import Trainer

from src.config import Experiment, ExperimentConfig, hydra_main
from src.trackers import get_trackers, save_learning_curves
from src.config.environment import DEBUG_MODE
from src.lib import get_learning_schema, get_loaders
from src.lib.hooks import register_checkpointing, register_early_stopping
from src.lib.objectives import get_classifier_loss
from src.lib.models import get_classifier_model
from src.utils.parallel import DistributedWorker


def train_classification() -> None:
    """Train the classifier and save it."""
    cfg = Experiment.get_config()
    user_cfg = cfg.user
    class_cfg = cfg.classification
    model = get_classifier_model()
    train_loader, eval_loader = get_loaders(
        batch_size=class_cfg.train.batch_size_per_device, n_workers=user_cfg.n_workers
    )
    builder = Trainer.builder(
        model,
        learning_schema=get_learning_schema(class_cfg),
        loader=train_loader,
        loss=get_classifier_loss(),
    )
    if not cfg.final:
        builder.add_validation(eval_loader)

    if not cfg.final and class_cfg.train.early_stopping.active:
        cfg_early = class_cfg.train.early_stopping
        register_early_stopping(builder, window=cfg_early.window, patience=cfg_early.patience)

    register_checkpointing(builder, checkpoint_every=user_cfg.checkpoint_every)

    trainer = builder.build()
    if user_cfg.load:
        trainer.load_checkpoint(user_cfg.load)

    trainer.train_until(class_cfg.train.n_epochs)
    trainer.save_checkpoint()
    return


def setup_and_train(cfg: ExperimentConfig) -> None:
    """Set up the experiment and train the classifier.

    Args:
        cfg: root configuration object.
    """
    trackers = get_trackers(cfg)
    resume = cfg.user.load != 0 or cfg.user.run_id is not None
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=resume, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        train_classification()
        save_learning_curves(trackers, cfg.classification.model.name, show=cfg.user.plot.show)

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
