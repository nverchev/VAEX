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

FACE_CENTRE = (0.48, 0.50)
"""Centre of the face ellipse in aligned CelebA, as fractions of the image."""
FACE_AXES = (0.42, 0.32)
"""Semi-axes of the face ellipse, as fractions of the image."""


def outside_face(size: int, device: torch.device) -> torch.Tensor:
    """Boolean mask selecting the region a counterfactual should leave alone."""
    rows = torch.linspace(0.0, 1.0, size, device=device)[:, None]
    cols = torch.linspace(0.0, 1.0, size, device=device)[None, :]
    inside = (
        ((rows - FACE_CENTRE[0]) / FACE_AXES[0]) ** 2
        + ((cols - FACE_CENTRE[1]) / FACE_AXES[1]) ** 2
    ) <= 1.0
    return ~inside


@torch.inference_mode()
def run() -> None:
    cfg = Experiment.get_config()
    module = load_extract_generative_module()
    classifier = load_extract_classifier_module()
    device = next(module.parameters()).device
    _, loader = get_loaders(cfg.autoencoder.train.batch_size_per_device, classifier=classifier)

    base = [1.0] * len(module.z_dims)
    n = len(base)
    conditions: dict[str, list[float]] = {'baseline': list(base)}
    for stage in range(n):
        zeroed = list(base)
        zeroed[stage] = 0.0
        conditions[f'stage {stage} -> 0'] = zeroed

    for level in (0.75, 0.5, 0.25, 0.0):
        conditions[f'all cond -> {level}'] = [level] * n

    conditions['graded'] = list(module.posterior_weights)

    totals = {name: 0.0 for name in conditions}
    context = {name: 0.0 for name in conditions}
    face = {name: 0.0 for name in conditions}
    n_seen = 0
    for i, (inputs, targets) in enumerate(loader):
        if i >= N_BATCHES:
            break

        inputs = Inputs(*(tensor.to(device) for tensor in inputs))
        label = targets.label.to(device)
        flipped = 1 - label
        recon = module(inputs).recon
        mask = outside_face(recon.size(-1), device)
        for name, weights in conditions.items():
            probs = []
            recons = []
            for target_dim in range(module.dim_target):
                out = module.generate_counterfactual(
                    inputs, target_dim=target_dim, posterior_weights=tuple(weights)
                )
                recons.append(out.recon)
                probs.append(torch.softmax(classifier(Inputs(image=out.recon)), dim=1))

            stacked = torch.stack(probs)  # (target, batch, class)
            chosen = stacked[flipped, torch.arange(label.numel(), device=device)]
            totals[name] += float(chosen.gather(1, flipped[:, None]).sum())
            images = torch.stack(recons)[flipped, torch.arange(label.numel(), device=device)]
            drift = (images - recon) ** 2
            context[name] += float(drift[:, :, mask].mean(dim=(1, 2)).sum())
            face[name] += float(drift[:, :, ~mask].mean(dim=(1, 2)).sum())

        n_seen += label.numel()

    logging.info('over %d samples: flip = P(target); drift = MSE vs reconstruction', n_seen)
    logging.info('%-16s %8s %12s %12s %10s', '', 'flip', 'face drift', 'context', 'ratio')
    for name in conditions:
        flip = totals[name] / n_seen
        ctx = context[name] / n_seen
        fac = face[name] / n_seen
        logging.info(
            '%-16s %8.4f %12.5f %12.5f %10.2f', name, flip, fac, ctx, fac / max(ctx, 1e-9)
        )

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        run()

    return None


if __name__ == '__main__':
    main()
