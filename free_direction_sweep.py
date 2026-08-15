"""Ask whether shifting the free half along its gender direction removes facial hair."""

import logging

import pandas as pd
import torch

from free_cut_sweep import BATCH, CHIN_AXES, CHIN_CENTRE, N_SAMPLES, ellipse
from inspect_latents import cond_widths
from stage_ablation import FACE_AXES, FACE_CENTRE, outside_face
from src.config import AllConfig, Experiment, hydra_main
from src.data.derived_datasets import EvaluatedDataset
from src.data.dataset import get_databuilder, get_datasets
from src.data.structures import Inputs
from src.train.models import load_extract_classifier_module, load_extract_generative_module

N_DIRECTION = 2000
"""Samples used to estimate the direction of the free half."""
USE_BEARD_DIRECTION = True
"""Whether the direction separates bearded from clean-shaven men rather than the classes."""
ALPHAS = (0.0, 2.0, 4.0, 8.0, 16.0, 32.0)
"""Multiples of the male-to-female shift applied to the free half."""


def batch_inputs(evaluated, index_list: list[int], device: torch.device) -> Inputs:
    """Collate a batch of evaluated samples onto the device."""
    pairs = evaluated.__getitems__(index_list)
    return Inputs(
        torch.stack([p[0].image for p in pairs]).to(device),
        torch.stack([p[0].condition for p in pairs]).to(device),
        torch.stack([p[0].log_probs for p in pairs]).to(device),
    )


@torch.inference_mode()
def run() -> None:
    module = load_extract_generative_module()
    device = next(module.parameters()).device
    classifier = load_extract_classifier_module()
    _, eval_dataset = get_datasets()
    evaluated = EvaluatedDataset(eval_dataset, classifier)

    extra = pd.read_csv(get_databuilder().data_dir / 'list_attr_celeba.txt', sep=r'\s+', skiprows=1, usecols=['No_Beard'])
    extra.reset_index(inplace=True)
    extra.columns = ['image_id', 'No_Beard']
    extra['image_id'] = extra['image_id'].astype(str).str.replace('.jpg', '', regex=False).astype(int)
    joined = eval_dataset.metadata.merge(extra, on='image_id', how='left')

    widths = cond_widths(module)
    is_beard = torch.tensor((joined['No_Beard'].to_numpy() == -1).astype(int), device=device)
    is_male = torch.tensor((joined['Male'].to_numpy() == 1).astype(int), device=device)
    pool = (is_male == 1).nonzero().flatten()[:N_DIRECTION] if USE_BEARD_DIRECTION else torch.arange(
        min(N_DIRECTION, len(joined)), device=device
    )
    sums: list[torch.Tensor] = []
    counts = torch.zeros(2, device=device)
    for start in range(0, len(pool), BATCH):
        index_list = pool[start: start + BATCH].tolist()
        inputs = batch_inputs(evaluated, index_list, device)
        label = is_beard[torch.tensor(index_list, device=device)] if USE_BEARD_DIRECTION else inputs.log_probs.argmax(1)
        out = module(inputs)
        for stage, d_mu in enumerate(out.d_mu):
            free = d_mu[:, widths[stage]:]
            if not sums or stage >= len(sums):
                sums.append(torch.zeros(2, free.size(1), device=device))

            sums[stage].index_add_(0, label, free.float())

        counts.index_add_(0, label, torch.ones_like(label, dtype=torch.float))

    direction = tuple((s[1] / counts[1] - s[0] / counts[0]) for s in sums)
    logging.info('direction norms per stage: %s', [f'{d.norm():.2f}' for d in direction])

    bearded = joined.index[(joined['No_Beard'] == -1) & (joined['Male'] == 1)].to_numpy()[:N_SAMPLES]
    size = module.x_dim
    chin = ellipse(size, device, CHIN_CENTRE, CHIN_AXES)
    face = ellipse(size, device, FACE_CENTRE, FACE_AXES)
    context = outside_face(size, device)
    rest = face & ~chin
    cut = tuple([0.0] * len(module.z_dims))

    logging.info('')
    logging.info('bearded men: %d | conditioned half fully cut, free half shifted', len(bearded))
    logging.info('%9s %9s %11s %11s %11s %11s', 'alpha', 'flip', 'chin drift', 'face drift', 'context', 'chin/ctx')
    for alpha in ALPHAS:
        shift = tuple(-alpha * d for d in direction)
        flips, chins, rests, contexts = [], [], [], []
        for start in range(0, len(bearded), BATCH):
            inputs = batch_inputs(evaluated, bearded[start: start + BATCH].tolist(), device)
            recon = module(inputs).recon
            target = int(1 - inputs.log_probs[0].argmax())
            out_cf = module.generate_counterfactual(
                inputs, target_dim=target, posterior_weights=cut, free_shift=shift
            )
            delta = (out_cf.recon - recon) ** 2
            probs = torch.softmax(classifier(Inputs(image=out_cf.recon)), dim=1)
            flips.append(probs[:, target].cpu())
            chins.append(delta[:, :, chin].mean((1, 2)).cpu())
            rests.append(delta[:, :, rest].mean((1, 2)).cpu())
            contexts.append(delta[:, :, context].mean((1, 2)).cpu())

        chin_drift = float(torch.cat(chins).mean())
        context_drift = float(torch.cat(contexts).mean())
        logging.info(
            '%9.2f %9.4f %11.5f %11.5f %11.5f %11.2f',
            alpha,
            float(torch.cat(flips).mean()),
            chin_drift,
            float(torch.cat(rests).mean()),
            context_drift,
            chin_drift / max(context_drift, 1e-12),
        )

    return None


@hydra_main
def main(cfg: AllConfig) -> None:
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True, record=False):
        run()

    return None


if __name__ == '__main__':
    main()
