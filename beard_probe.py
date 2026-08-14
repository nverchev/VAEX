"""Ask where facial-hair information lives and whether the counterfactual removes it."""

import logging

import pandas as pd
import torch

from inspect_latents import cond_widths, probe
from src.config import AllConfig, Experiment, hydra_main
from src.data.derived_datasets import EvaluatedDataset
from src.data.dataset import get_databuilder, get_datasets
from src.data.structures import Inputs
from src.train.models import load_extract_autoencoder_module, load_extract_classifier_module, load_extract_generative_module

N_SAMPLES = 4000
"""Evaluation samples drawn for every table below."""
BATCH = 32
"""Samples per forward pass."""
HIDDEN = 512
"""Probe width matching the eraser, marked with a star in the tables."""


def balanced_probe(features: torch.Tensor, labels: torch.Tensor, hidden: int = 0) -> float:
    """Probe accuracy after equalising the two classes, so the base rate is one half."""
    pos = (labels == 1).nonzero().flatten()
    neg = (labels == 0).nonzero().flatten()
    k = min(pos.numel(), neg.numel())
    keep = torch.cat([pos[:k], neg[:k]])
    keep = keep[torch.randperm(keep.numel(), generator=torch.Generator().manual_seed(0))]
    return probe(features[keep], labels[keep], hidden=hidden)


@torch.no_grad()
def report() -> None:
    module = load_extract_autoencoder_module()
    generative = load_extract_generative_module()
    device = next(module.parameters()).device
    classifier = load_extract_classifier_module()
    _, eval_dataset = get_datasets()
    metadata = eval_dataset.metadata
    evaluated = EvaluatedDataset(eval_dataset, classifier)

    extra = pd.read_csv(get_databuilder().data_dir / 'list_attr_celeba.txt', sep=r'\s+', skiprows=1, usecols=['No_Beard'])
    extra.reset_index(inplace=True)
    extra.columns = ['image_id', 'No_Beard']
    extra['image_id'] = extra['image_id'].astype(str).str.replace('.jpg', '', regex=False).astype(int)
    joined = metadata.merge(extra, on='image_id', how='left')

    n = min(N_SAMPLES, len(joined))
    order = torch.randperm(len(joined), generator=torch.Generator().manual_seed(0))[:n]
    beard = torch.tensor((joined['No_Beard'].to_numpy()[order.numpy()] == -1).astype(int))
    male = torch.tensor((joined['Male'].to_numpy()[order.numpy()] == 1).astype(int))

    means: list[list[torch.Tensor]] = []
    confidence: list[torch.Tensor] = []
    flipped_full: list[torch.Tensor] = []
    for start in range(0, n, BATCH):
        index_list = order[start: start + BATCH].tolist()
        pairs = evaluated.__getitems__(index_list)
        image = torch.stack([p[0].image for p in pairs]).to(device)
        condition = torch.stack([p[0].condition for p in pairs]).to(device)
        log_probs = torch.stack([p[0].log_probs for p in pairs]).to(device)
        inputs = Inputs(image, condition, log_probs)

        out = module(inputs)
        for stage, (p_mu, d_mu) in enumerate(zip(out.prior_mu, out.d_mu, strict=True)):
            if start == 0:
                means.append([])

            means[stage].append((p_mu + d_mu).float().cpu())

        label = log_probs.argmax(1)
        confidence.append(log_probs.exp().max(1).values.cpu())

        target = 1 - label
        recon_cf = torch.stack([
            generative.generate_counterfactual(
                Inputs(image[i: i + 1], condition[i: i + 1], log_probs[i: i + 1]),
                target_dim=int(target[i]),
                posterior_weights=tuple([0.0] * len(module.z_dims)),
            ).recon[0]
            for i in range(image.size(0))
        ])
        cf_probs = torch.softmax(classifier(Inputs(image=recon_cf)), dim=1).cpu()
        flipped_full.append(cf_probs.gather(1, target.cpu()[:, None]).squeeze(1))

    conf = torch.cat(confidence)
    flip = torch.cat(flipped_full)
    widths = cond_widths(module)

    logging.info('')
    logging.info('facial hair in the eval sample: %d of %d (%.1f%%)', int(beard.sum()), n, 100 * beard.float().mean())
    logging.info('')
    logging.info('does the classifier need the beard? (mean confidence in its own call)')
    for name, mask in (('bearded men', male.bool() & beard.bool()), ('clean-shaven men', male.bool() & ~beard.bool())):
        logging.info('  %-20s %.4f  (n = %d)', name, float(conf[mask].mean()), int(mask.sum()))

    logging.info('')
    logging.info('does the counterfactual work on them? (P(target) after full cut)')
    for name, mask in (('bearded men', male.bool() & beard.bool()), ('clean-shaven men', male.bool() & ~beard.bool())):
        logging.info('  %-20s %.4f', name, float(flip[mask].mean()))

    base_male = float(max(male.float().mean(), 1 - male.float().mean()))
    logging.info('')
    logging.info('where does facial hair live? (probe on mu, beard base 0.500 balanced; gender base %.3f)', base_male)
    logging.info(
        '%5s %7s %7s %10s %10s %10s %10s %10s',
        'stage', 'cond', 'free', 'beard-c', 'beard-f', 'beard-f*', 'gender-f', 'gender-f*',
    )
    for stage, parts in enumerate(means):
        mu = torch.cat(parts)
        split = widths[stage] if stage < len(widths) else 0
        if not split or split >= mu.size(1):
            logging.info(
                '%5d %7d %7s %10.3f %10s %10s %10s %10s',
                stage, mu.size(1), '-', balanced_probe(mu, beard), '-', '-', '-', '-',
            )
            continue

        logging.info(
            '%5d %7d %7d %10.3f %10.3f %10.3f %10.3f %10.3f',
            stage, split, mu.size(1) - split,
            balanced_probe(mu[:, :split], beard),
            balanced_probe(mu[:, split:], beard),
            balanced_probe(mu[:, split:], beard, hidden=HIDDEN),
            probe(mu[:, split:], male),
            probe(mu[:, split:], male, hidden=HIDDEN),
        )

    return None


@hydra_main
def main(cfg: AllConfig) -> None:
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True, record=False):
        report()

    return None


if __name__ == '__main__':
    main()
