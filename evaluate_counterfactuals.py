"""Evaluate the VAE, the success of its counterfactuals for each posterior relaxation, and the FID.

Usage::

    uv run evaluate_counterfactuals.py
    uv run evaluate_counterfactuals.py final=true
"""

import logging
from collections.abc import Iterator, Mapping, Sequence, Sized
from typing import cast

import torch
from torch import Tensor

from drytorch import DataLoader, Test
from drytorch.lib import loading
from drytorch.lib.objectives import Metric, compute_metrics

from src.config import Experiment, ExperimentConfig, hydra_main
from src.trackers import get_trackers
from src.data.dataset import get_split
from src.data.derived_datasets import CounterfactualDataset, EvaluatedDataset, ReconstructedEvaluatedDataset
from src.data.split import DataSplit, Partitions, Subset
from src.data.structures import Inputs, Targets
from src.utils.fid import FrechetInceptionDistance
from src.lib.objectives import get_classifier_loss, get_counterfactual_success, get_vae_loss
from src.lib.models import ClassifierModel, VAEModel, load_autoencoder_model, get_classifier_model, load_trained_state


def evaluate_success(
    classifier_model: ClassifierModel,
    split: DataSplit,
    vae_model: VAEModel,
    groups: Mapping[int, Sequence[int]],
    relaxation: float,
    batch_size: int,
    *,
    name: str,
) -> float:
    """Get the fraction of counterfactuals that the classifier assigns to their target.

    Args:
        classifier_model: the classifier.
        split: the images.
        vae_model: the trained VAE.
        groups: the indices of the images turned towards each target class.
        relaxation: weight of the posterior of the lower latents.
        batch_size: number of images in a batch.
        name: the prefix of the names of the tests.

    Returns:
        The fraction of successful counterfactuals.
    """
    metrics: list[Metric[Tensor, Targets]] = []
    for target, indices in groups.items():
        if not indices:
            continue

        counterfactuals = CounterfactualDataset(
            Subset(split, indices),
            autoencoder=vae_model,
            classifier=classifier_model,
            target_class=target,
            relaxation=relaxation,
        )
        loader = DataLoader(dataset=counterfactuals, batch_size=batch_size, pin_memory=False)
        test_name = f'{name}_r{relaxation:.2f}_to_{target}'
        success = get_counterfactual_success()
        Test(classifier_model, name=test_name, loader=loader, metric=success)()
        metrics.append(success)

    overall = metrics[0].copy()
    for metric in metrics[1:]:
        overall.merge_state(metric)

    return float(compute_metrics(overall)['Success'])


def evaluate_fid(
    classifier_model: ClassifierModel,
    vae_model: VAEModel,
    split: DataSplit,
    predictions: Tensor,
    batch_size: int,
) -> None:
    """Log the FID of the reconstructions and of the counterfactuals.

    The first ``evaluation.fid_samples`` images are reconstructed and turned towards a
    class the classifier does not predict for them, with the posterior weighted by
    ``evaluation.fid_relaxation``; the next as many images are the reference.

    Args:
        classifier_model: the trained classifier.
        vae_model: the trained VAE.
        split: the images of the evaluation partition.
        predictions: the classes the classifier predicts for the images.
        batch_size: number of images in a batch.

    Raises:
        ValueError: if the partition has fewer than twice ``fid_samples`` images.
    """
    n_samples = Experiment.get_config().evaluation.fid_samples
    if 2 * n_samples > len(cast(Sized, split)):
        msg = f'The FID needs {2 * n_samples} images, the evaluation partition has {len(cast(Sized, split))}.'
        raise ValueError(msg)

    device = vae_model.device
    reference = _take_batches(split, range(n_samples, 2 * n_samples), batch_size, device)
    logging.info('Calculating the Inception features of %d reference images.', n_samples)
    fid = FrechetInceptionDistance((inputs.image for inputs in reference), device=device)
    reconstructions = _reconstruct(classifier_model, vae_model, split, range(n_samples), batch_size)
    distance = fid(vae_model, reconstructions, name='FID_reconstructions')
    logging.info('FID of the reconstructions: %.1f', distance)
    relaxation = Experiment.get_config().evaluation.fid_relaxation
    images = _generate_counterfactuals(
        classifier_model, vae_model, split, predictions[:n_samples], relaxation, batch_size
    )
    distance = fid(vae_model, images, name=f'FID_counterfactuals_r{relaxation:.2f}')
    logging.info('FID of the counterfactuals (r=%.2f): %.1f', relaxation, distance)
    return


@torch.inference_mode()
def evaluate_counterfactuals() -> None:
    """Evaluate the trained classifier, the reconstructions, the classifier on them, and the counterfactuals."""
    cfg = Experiment.get_config()
    classifier_model = get_classifier_model()
    load_trained_state(classifier_model)
    vae_model = load_autoencoder_model()
    batch_size = cfg.autoencoder.train.batch_size
    split = get_split(Partitions.test if cfg.final else Partitions.val)
    loader = DataLoader(dataset=split, batch_size=batch_size, pin_memory=False)
    original = Test(classifier_model, name='ClassificationOriginal', loader=loader, metric=get_classifier_loss())
    original(store_outputs=True)
    predictions = torch.cat(list(original.outputs_list)).argmax(dim=1).cpu()
    evaluated = EvaluatedDataset(split, classifier_model)
    evaluated_loader = DataLoader(dataset=evaluated, batch_size=batch_size, pin_memory=False)
    Test(vae_model, name='Reconstruction', loader=evaluated_loader, metric=get_vae_loss())()
    reconstructed = ReconstructedEvaluatedDataset(split, autoencoder=vae_model, classifier=classifier_model)
    reconstructed_loader = DataLoader(dataset=reconstructed, batch_size=batch_size, pin_memory=False)
    Test(
        classifier_model,
        name='Classification_reconstructions',
        loader=reconstructed_loader,
        metric=get_classifier_loss(),
    )()
    labels = torch.cat([targets.label for _, targets in loader]).cpu()
    classes = range(vae_model.module.n_classes)
    groups = {target: _get_indices(predictions != target) for target in classes}
    misclassified = {target: _get_indices((labels == target) & (predictions != target)) for target in classes}
    for relaxation in cfg.evaluation.relaxations:
        success = evaluate_success(
            classifier_model, split, vae_model, groups, relaxation, batch_size, name='Counterfactual'
        )
        logging.info('Successful counterfactuals with r=%.2f: %.1f%%', relaxation, 100 * success)
        success = evaluate_success(
            classifier_model,
            split,
            vae_model,
            misclassified,
            relaxation,
            batch_size,
            name='Counterfactual_misclassified',
        )
        logging.info('Misclassified turned towards their label with r=%.2f: %.1f%%', relaxation, 100 * success)

    evaluate_fid(classifier_model, vae_model, split, predictions, batch_size)
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and launch the evaluation."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        evaluate_counterfactuals()

    return


@torch.inference_mode()
def _generate_counterfactuals(
    classifier_model: ClassifierModel,
    vae_model: VAEModel,
    split: DataSplit,
    predictions: Tensor,
    relaxation: float,
    batch_size: int,
) -> Iterator[Tensor]:
    for target in range(vae_model.module.n_classes):  # consider only images with different predictions
        indices = (predictions != target).nonzero().squeeze(-1).tolist()
        if not indices:
            continue

        for inputs in _take_batches(split, indices, batch_size, vae_model.device):
            inputs = inputs._replace(log_probs=classifier_model.log_probabilities(inputs))
            yield vae_model.generate_counterfactual(inputs, target, relaxation).recon

    return


@torch.inference_mode()
def _reconstruct(
    classifier_model: ClassifierModel,
    vae_model: VAEModel,
    split: DataSplit,
    indices: Sequence[int],
    batch_size: int,
) -> Iterator[Tensor]:
    for inputs in _take_batches(split, indices, batch_size, vae_model.device):
        labelled = inputs._replace(log_probs=classifier_model.log_probabilities(inputs))
        yield vae_model.reconstruct(labelled).recon

    return


def _take_batches(
    split: DataSplit,
    indices: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> Iterator[Inputs]:
    for start in range(0, len(indices), batch_size):
        _, (inputs, _) = loading.take_from_dataset(split, indices=indices[start : start + batch_size], device=device)
        yield inputs

    return


def _get_indices(mask: Tensor) -> list[int]:
    return mask.nonzero().squeeze(1).tolist()


if __name__ == '__main__':
    main()
