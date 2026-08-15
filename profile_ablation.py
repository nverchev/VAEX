"""Compare posterior-cut profiles: schedules against subsets of stages."""

import logging

import torch

from stage_ablation import N_BATCHES, outside_face
from src.config import AllConfig, Experiment, hydra_main
from src.data.structures import Inputs
from src.train.loaders import get_loaders
from src.train.models import load_extract_classifier_module, load_extract_generative_module


def build_profiles(n: int, module) -> dict[str, list[float]]:
    """Named per-stage weight vectors spanning schedules and subsets."""
    profiles: dict[str, list[float]] = {'baseline': [1.0] * n}
    for level in (0.5, 0.25, 0.0):
        profiles[f'constant {level}'] = [level] * n

    profiles['linear coarse-cut'] = [step / (n - 1) for step in range(n)]
    profiles['linear fine-cut'] = [1.0 - step / (n - 1) for step in range(n)]
    profiles['exponential'] = list(module.posterior_weights_from(0.5))
    for subset in ((3,), (4,), (3, 4), (2, 3, 4), (1, 2, 3, 4), (2, 3, 4, 5)):
        weights = [1.0] * n
        for stage in subset:
            weights[stage] = 0.0

        profiles['cut ' + '+'.join(map(str, subset))] = weights

    return profiles


@torch.inference_mode()
def run() -> None:
    cfg = Experiment.get_config()
    module = load_extract_generative_module()
    classifier = load_extract_classifier_module()
    device = next(module.parameters()).device
    _, loader = get_loaders(cfg.autoencoder.train.batch_size_per_device, classifier=classifier)

    profiles = build_profiles(len(module.z_dims), module)
    flip = {name: 0.0 for name in profiles}
    face = {name: 0.0 for name in profiles}
    context = {name: 0.0 for name in profiles}
    n_seen = 0
    for i, (inputs, targets) in enumerate(loader):
        if i >= N_BATCHES:
            break

        inputs = Inputs(*(tensor.to(device) for tensor in inputs))
        label = targets.label.to(device)
        flipped = 1 - label
        recon = module(inputs).recon
        mask = outside_face(recon.size(-1), device)
        for name, weights in profiles.items():
            probs, recons = [], []
            for target_dim in range(module.dim_target):
                out = module.generate_counterfactual(
                    inputs, target_dim=target_dim, posterior_weights=tuple(weights)
                )
                recons.append(out.recon)
                probs.append(torch.softmax(classifier(Inputs(image=out.recon)), dim=1))

            index = torch.arange(label.numel(), device=device)
            chosen = torch.stack(probs)[flipped, index]
            cf = torch.stack(recons)[flipped, index]
            delta = (cf - recon) ** 2
            flip[name] += float(chosen[index, flipped].sum())
            face[name] += float(delta[:, :, ~mask].mean((1, 2)).sum())
            context[name] += float(delta[:, :, mask].mean((1, 2)).sum())

        n_seen += label.numel()

    logging.info('over %d samples', n_seen)
    logging.info('%-20s %9s %11s %11s %9s', 'profile', 'flip', 'face drift', 'context', 'ratio')
    for name in profiles:
        f, c = face[name] / n_seen, context[name] / n_seen
        logging.info('%-20s %9.4f %11.5f %11.5f %9.2f', name, flip[name] / n_seen, f, c, f / max(c, 1e-12))

    return None


@hydra_main
def main(cfg: AllConfig) -> None:
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True, record=False):
        run()

    return None


if __name__ == '__main__':
    main()
