"""Zero one stage's posterior at a time and see which one is blocking the flip.

An intervention keeps whatever the posterior weights retain, so a stage holding class
evidence shows up as a jump in the target probability when its posterior is removed.
"""

import logging

import torch

from src.config import AllConfig, Experiment, hydra_main
from src.data.structures import Inputs
from src.train.loaders import get_loaders
from src.train.models import load_extract_classifier_module, load_extract_generative_module

N_BATCHES = 16


@torch.inference_mode()
def run() -> None:
    cfg = Experiment.get_config()
    module = load_extract_generative_module()
    classifier = load_extract_classifier_module()
    device = next(module.parameters()).device
    _, loader = get_loaders(cfg.autoencoder.train.batch_size_per_device, classifier=classifier)

    # Start from the full posterior so each stage is removed from the same footing.
    base = [1.0] * len(module.z_dims)
    n = len(base)
    conditions: dict[str, list[float]] = {'baseline': list(base)}
    for stage in range(n):
        zeroed = list(base)
        zeroed[stage] = 0.0
        conditions[f'stage {stage} -> 0'] = zeroed

    # All conditioned halves cut together, which is what the split makes affordable.
    for level in (0.75, 0.5, 0.25, 0.0):
        conditions[f'all cond -> {level}'] = [level] * n

    conditions['graded'] = list(module.posterior_weights)

    totals = {name: 0.0 for name in conditions}
    n_seen = 0
    for i, (inputs, targets) in enumerate(loader):
        if i >= N_BATCHES:
            break

        inputs = Inputs(*(tensor.to(device) for tensor in inputs))
        label = targets.label.to(device)
        flipped = 1 - label
        for name, weights in conditions.items():
            probs = []
            for target_dim in range(module.dim_target):
                out = module.generate_counterfactual(
                    inputs, target_dim=target_dim, posterior_weights=tuple(weights)
                )
                probs.append(torch.softmax(classifier(Inputs(image=out.recon)), dim=1))

            stacked = torch.stack(probs)  # (target, batch, class)
            chosen = stacked[flipped, torch.arange(label.numel(), device=device)]
            totals[name] += float(chosen.gather(1, flipped[:, None]).sum())

        n_seen += label.numel()

    logging.info('flip target probability over %d samples (higher = stronger counterfactual)', n_seen)
    baseline = totals['baseline'] / n_seen
    for name, total in totals.items():
        mean = total / n_seen
        logging.info('%-16s %.4f   %+.4f', name, mean, mean - baseline)

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        run()

    return None


if __name__ == '__main__':
    main()
