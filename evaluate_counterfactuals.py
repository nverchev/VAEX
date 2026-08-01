"""Evaluate counterfactual generation performance."""

import logging
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset

from drytorch import DataLoader, Model, Test
from drytorch.core import protocols as p
from drytorch.lib.objectives import Metric, compute_metrics

from src.config import AllConfig, Experiment, hydra_main
from src.data.derived_datasets import CounterfactualDataset, ReconstructedEvaluatedDataset
from src.data.dataset import get_dataset
from src.data.split import Partitions
from src.data.structures import Inputs, Targets
from src.module.autoencoder import Abstract_VAE
from src.module.classifier import BaseClassifier, get_classifier
from src.train.metrics_and_losses import get_cross_entropy
from src.train.models import load_extract_autoencoder_module


def get_label_distribution(test_loader: p.LoaderProtocol[tuple[Inputs, Targets]], class_names: list[str]) -> Tensor:
    """Extract labels from the test loader and compute distribution."""
    with torch.inference_mode():
        labels = torch.cat([batch_data[1].label for batch_data in test_loader])

    counts = [(labels == i).sum().item() for i in range(len(class_names))]
    dist_str = ' | '.join(f'{name}: {int(count)}' for name, count in zip(class_names, counts, strict=True))
    logging.info('Label distribution: %s', dist_str)
    return labels


def evaluate_original_performance(
    classifier_model: Model[Inputs, Tensor],
    test_loader: DataLoader[tuple[Inputs, Targets]],
) -> Test[Inputs, Targets, Tensor]:
    """Evaluate classifier performance on original test data."""
    loss = get_cross_entropy()
    test_original = Test(classifier_model, name='ClassificationOriginal', loader=test_loader, metric=loss)
    test_original(store_outputs=True)
    return test_original


def evaluate_reconstructed_performance(
    classifier_model: Model[Inputs, Tensor],
    test_dataset: Dataset[tuple[Inputs, Targets]],
    vae: Abstract_VAE,
    classifier: BaseClassifier,
    batch_size: int,
) -> None:
    """Evaluate classifier performance on reconstructed test data."""
    reconstructed_dataset = ReconstructedEvaluatedDataset(
        dataset=test_dataset,
        autoencoder=vae,
        classifier=classifier,
    )
    reconstructed_loader = DataLoader(dataset=reconstructed_dataset, batch_size=batch_size, pin_memory=False)
    loss = get_cross_entropy()
    test_reconstructed = Test(
        classifier_model, name='ClassificationReconstructed', loader=reconstructed_loader, metric=loss
    )
    test_reconstructed()
    return None


def evaluate_counterfactual_performance(
    classifier_model: Model[Inputs, Tensor],
    test_dataset: Dataset[tuple[Inputs, Targets]],
    vae: Abstract_VAE,
    classifier: BaseClassifier,
    class_names: list[str],
    batch_size: int,
) -> None:
    """Evaluate counterfactual generation performance across all classes."""
    metrics: list[Metric[Tensor, Targets]] = []
    loss = get_cross_entropy()

    for j, class_name in enumerate(class_names):
        counterfactual_dataset = CounterfactualDataset(
            test_dataset, autoencoder=vae, classifier=classifier, target_class=j
        )
        counterfactual_loader = DataLoader(dataset=counterfactual_dataset, batch_size=batch_size, pin_memory=False)
        test = Test(classifier_model, name=f'Counterfeit_to_{class_name}', loader=counterfactual_loader, metric=loss)
        test()
        metrics.append(cast(Metric[Tensor, Targets], test.objective))

    overall_metric = compute_overall_metric(metrics)
    if overall_metric:
        logging.info('Overall counterfeit success:')
        log_metrics(overall_metric)

    return None


def evaluate_misclassified_samples(
    classifier_model: Model[Inputs, Tensor],
    test_dataset: Dataset[tuple[Inputs, Targets]],
    vae: Abstract_VAE,
    classifier: BaseClassifier,
    labels: Tensor,
    predictions: Tensor,
    batch_size: int,
) -> None:
    """Evaluate reconstruction performance on misclassified samples."""
    labels = labels.to(predictions.device)
    misclassified_bool = predictions != labels

    misclassified_indices = misclassified_bool.nonzero().squeeze(-1)
    if misclassified_indices.numel() == 0:
        logging.info('No misclassified samples found.')
        return None

    misclassified_dataset = Subset(test_dataset, indices=list(map(int, misclassified_indices)))
    misclassified_reconstruction = ReconstructedEvaluatedDataset(
        dataset=misclassified_dataset, autoencoder=vae, classifier=classifier
    )
    misclassified_reconstruction_loader = DataLoader(
        misclassified_reconstruction, batch_size=batch_size, pin_memory=False
    )
    loss = get_cross_entropy()
    test_misclassified = Test(
        classifier_model, name='MisclassifiedReconstructed', loader=misclassified_reconstruction_loader, metric=loss
    )
    test_misclassified()
    return None


def evaluate_class_specific_counterfactuals(
    classifier_model: Model[Inputs, Tensor],
    test_dataset: Dataset[tuple[Inputs, Targets]],
    vae: Abstract_VAE,
    classifier: BaseClassifier,
    labels: Tensor,
    predictions: Tensor,
    class_names: list[str],
    batch_size: int,
) -> None:
    """Evaluate counterfactual performance for specific class transitions."""
    labels = labels.to(predictions.device)
    metrics: list[Metric[Tensor, Targets]] = []
    loss = get_cross_entropy()

    for i, name_i in enumerate(class_names):
        for j, name_j in enumerate(class_names):
            if i == j:
                continue

            i_instead_of_j = torch.logical_and(predictions == i, labels == j)
            if not torch.any(i_instead_of_j):
                continue

            counterfactual_indices = list(i_instead_of_j.nonzero().squeeze(-1))
            i_instead_of_j_dataset = Subset(test_dataset, indices=list(map(int, counterfactual_indices)))
            counterfactual_dataset = CounterfactualDataset(
                i_instead_of_j_dataset,
                autoencoder=vae,
                classifier=classifier,
                target_class=j,
            )
            counterfactual_loader = DataLoader(dataset=counterfactual_dataset, batch_size=batch_size, pin_memory=False)
            test = Test(classifier_model, name=f'{name_i}_to_{name_j}', loader=counterfactual_loader, metric=loss)
            test(store_outputs=True)
            metrics.append(cast(Metric[Tensor, Targets], test.objective))

    if metrics:
        overall_metric = compute_overall_metric(metrics)
        if overall_metric:
            logging.info('Overall misclassified counterfeit success:')
            log_metrics(overall_metric)

    return None


def log_metrics(metrics: p.ObjectiveProtocol[Tensor, Targets], prefix: str = '') -> None:
    """Log metrics summary."""
    results = compute_metrics(metrics)
    metrics_str = ' | '.join(f'{k}: {v:.3f}' for k, v in results.items())
    logging.info('%s%s', prefix, metrics_str)
    return None


def compute_overall_metric(metrics: list[Metric[Tensor, Targets]]) -> Metric[Tensor, Targets] | None:
    """Compute overall metric by merging individual metrics."""
    if not metrics:
        return None

    overall_metric = metrics[0].copy()
    for metric in metrics[1:]:
        overall_metric.merge_state(metric)

    return overall_metric


@torch.inference_mode()
def evaluate_counterfactuals(
    classifier_model: Model[Inputs, Tensor],
    vae: Abstract_VAE,
    classifier: BaseClassifier,
) -> None:
    """Evaluate counterfactuals according to accuracy, reconstruction, and flip metrics."""
    cfg = Experiment.get_config()
    batch_size = cfg.classification.train.batch_size

    partition = Partitions.test if cfg.final else Partitions.val
    test_dataset = get_dataset(partition)
    class_names = cfg.data.dataset.classes
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    test_labels = get_label_distribution(test_loader, class_names)
    test_original = evaluate_original_performance(classifier_model, test_loader)
    evaluate_reconstructed_performance(classifier_model, test_dataset, vae, classifier, batch_size)
    evaluate_counterfactual_performance(classifier_model, test_dataset, vae, classifier, class_names, batch_size)

    outputs_logits = torch.cat(test_original.outputs_list)
    predictions = outputs_logits.argmax(dim=1)

    evaluate_misclassified_samples(
        classifier_model, test_dataset, vae, classifier, test_labels, predictions, batch_size
    )
    evaluate_class_specific_counterfactuals(
        classifier_model,
        test_dataset,
        vae,
        classifier,
        test_labels,
        predictions,
        class_names,
        batch_size,
    )
    return None


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch counterfactual evaluation."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        classifier = get_classifier().eval()

        classifier_model = Model(classifier, name=cfg.classification.name, device=cfg.user.device)
        try:
            classifier_model.load_state()
        except Exception:
            logging.warning('Could not load classifier weights; running evaluation with default model parameters.')

        ema_module = load_extract_autoencoder_module()
        if not isinstance(ema_module, Abstract_VAE):
            raise TypeError('EMA module is not an Abstract_VAE.')

        evaluate_counterfactuals(classifier_model, ema_module, classifier)

    return None


if __name__ == '__main__':
    main()
